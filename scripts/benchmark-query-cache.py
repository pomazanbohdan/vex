from __future__ import annotations

import argparse
import ctypes
import datetime
import hashlib
import json
import os
import platform
import shutil
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


MODEL_REPO = "models--Qdrant--all-MiniLM-L6-v2-onnx"
MODEL_FILES = (
    "model.onnx",
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
)
QUERY = "retrieve navigation menu metadata from native application source"
QUERY_VARIANT = "Retrieve navigation menu metadata from native application source"


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_snapshot(root: Path, excluded_top_level: set[str] | None = None) -> dict[str, str]:
    excluded = excluded_top_level or set()
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or (relative.parts and relative.parts[0] in excluded):
            continue
        files[relative.as_posix()] = sha256_file(path)
    return files


def prepare_native_fixture(project: Path) -> None:
    fixture = Path(__file__).resolve().parents[1] / "benchmarks" / "query-cache" / "fixture"
    shutil.copytree(fixture, project)


def copy_model_cache(source_root: Path, target_root: Path) -> dict[str, Any]:
    source_repo = source_root / MODEL_REPO
    if not source_repo.is_dir():
        raise FileNotFoundError(f"MiniLM cache repo is missing: {source_repo}")
    shutil.copytree(source_repo, target_root / MODEL_REPO, copy_function=shutil.copy2)
    copied = [path for path in (target_root / MODEL_REPO).rglob("*") if path.is_file()]
    return {
        "source": str(source_repo),
        "target": str(target_root / MODEL_REPO),
        "files": len(copied),
        "bytes": sum(path.stat().st_size for path in copied),
    }


def active_model_files(embeddings: Path) -> tuple[str, dict[str, Path]]:
    repo = embeddings / MODEL_REPO
    raw_revision = (repo / "refs" / "main").read_bytes()
    if len(raw_revision) != 40 or any(chr(byte) not in "0123456789abcdefABCDEF" for byte in raw_revision):
        raise ValueError("MiniLM refs/main is not one 40-digit hex revision")
    revision = raw_revision.decode("ascii")
    snapshot = repo / "snapshots" / revision
    paths = {name: snapshot / name for name in MODEL_FILES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MiniLM snapshot is incomplete: {missing}")
    return revision, paths


def cache_snapshot(cache_dir: Path) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    if not cache_dir.is_dir():
        return snapshot
    for path in sorted(cache_dir.glob("*.bin")):
        raw = path.read_bytes()
        entry: dict[str, Any] = {
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
            "mtime_ns": path.stat().st_mtime_ns,
        }
        if len(raw) >= 20 and raw[:4] == b"VEXE":
            id_length = struct.unpack_from("<I", raw, 8)[0]
            if 20 + id_length <= len(raw):
                entry["identity"] = raw[12 : 12 + id_length].decode("utf-8", errors="replace")
                entry["dim"], entry["count"] = struct.unpack_from("<II", raw, 12 + id_length)
        snapshot[path.name] = entry
    return snapshot


def query_cache_file(cache_dir: Path, query: str) -> Path:
    suffix = ":" + sha256_bytes(query.encode("utf-8"))
    matches = [
        cache_dir / name
        for name, entry in cache_snapshot(cache_dir).items()
        if entry.get("identity", "").endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one cache entry for exact query, found {len(matches)}")
    return matches[0]


def system_cpu_times() -> tuple[int, int] | None:
    if os.name != "nt":
        return None
    idle = ctypes.c_ulonglong()
    kernel = ctypes.c_ulonglong()
    user = ctypes.c_ulonglong()
    ok = ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
    )
    if not ok:
        return None
    return idle.value, kernel.value + user.value


def system_cpu_busy_percent(before: tuple[int, int] | None, after: tuple[int, int] | None) -> float | None:
    if before is None or after is None:
        return None
    idle_delta = after[0] - before[0]
    total_delta = after[1] - before[1]
    if total_delta <= 0:
        return None
    return 100.0 * (total_delta - idle_delta) / total_delta


class Recorder:
    def __init__(self, report_path: Path, report: dict[str, Any], env: dict[str, str], cache_dir: Path):
        self.report_path = report_path
        self.report = report
        self.env = env
        self.cache_dir = cache_dir

    def save(self) -> None:
        write_json(self.report_path, self.report)

    def run(
        self,
        kind: str,
        side: str,
        argv: list[str],
        cwd: Path,
        iteration: int | None = None,
        timeout: int = 300,
    ) -> dict[str, Any]:
        before_cache = cache_snapshot(self.cache_dir)
        started = time.perf_counter_ns()
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
            exit_code: int | str = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
            error = None
        except subprocess.TimeoutExpired as failure:
            exit_code = "timeout"
            stdout = failure.stdout or b""
            stderr = failure.stderr or b""
            error = f"timeout after {timeout}s"
        elapsed_ns = time.perf_counter_ns() - started
        decoded_stdout = stdout.decode("utf-8", errors="replace")
        decoded_stderr = stderr.decode("utf-8", errors="replace")
        data = None
        json_error = None
        if decoded_stdout:
            try:
                data = json.loads(decoded_stdout)
            except json.JSONDecodeError as failure:
                json_error = str(failure)
        results = data.get("results") if isinstance(data, dict) else None
        envelope_valid = (
            isinstance(data, dict)
            and data.get("protocol_version") == "v1"
            and isinstance(data.get("capabilities"), dict)
            and isinstance(data.get("_meta"), dict)
            and isinstance(results, (dict, list))
        )
        is_search = "search" in argv
        search_results_valid = envelope_valid and isinstance(results, list) and bool(results) and all(
            isinstance(result, dict) and isinstance(result.get("name"), str)
            and isinstance(result.get("path"), str) and isinstance(result.get("line"), int)
            for result in results
        )
        results_bytes = json.dumps(results, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        stable_envelope = json.loads(json.dumps(data)) if isinstance(data, dict) else data
        if isinstance(stable_envelope, dict):
            stable_envelope.get("_meta", {}).pop("vex.dev/index_age_ms", None)
        stable_bytes = json.dumps(stable_envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        row = {
            "kind": kind,
            "side": side,
            "iteration": iteration,
            "timestamp_utc": utc_now(),
            "argv": argv,
            "cwd": str(cwd),
            "elapsed_seconds": elapsed_ns / 1_000_000_000,
            "exit_code": exit_code,
            "error": error,
            "stdout": decoded_stdout,
            "stderr": decoded_stderr,
            "stdout_sha256": sha256_bytes(stdout),
            "stable_envelope_sha256": sha256_bytes(stable_bytes) if envelope_valid else None,
            "results_sha256": sha256_bytes(results_bytes) if envelope_valid else None,
            "envelope_valid": envelope_valid,
            "search_results_valid": search_results_valid,
            "json_error": json_error,
            "cache_before": before_cache,
            "cache_after": cache_snapshot(self.cache_dir),
        }
        self.report["events"].append(row)
        self.save()
        if is_search and (not successful(row) or not search_results_valid):
            raise RuntimeError(f"invalid search output: {kind}/{side}")
        print(
            json.dumps(
                {
                    "kind": kind,
                    "side": side,
                    "iteration": iteration,
                    "seconds": row["elapsed_seconds"],
                    "exit_code": exit_code,
                }
            ),
            flush=True,
        )
        return row


def successful(row: dict[str, Any]) -> bool:
    return row["exit_code"] == 0 and row["json_error"] is None and row["envelope_valid"]


def summarize_times(rows: list[dict[str, Any]], side: str) -> dict[str, Any]:
    times = [
        row["elapsed_seconds"]
        for row in rows
        if row["side"] == side and row["exit_code"] == 0
    ]
    return {
        "n": len(times),
        "median_seconds": statistics.median(times) if times else None,
        "range_seconds": [min(times), max(times)] if times else None,
        "times_seconds": times,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-exe", required=True, type=Path)
    parser.add_argument("--head-exe", required=True, type=Path)
    parser.add_argument("--model-cache-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs < 5:
        raise ValueError("--runs must be at least 5")
    baseline = args.baseline_exe.resolve(strict=True)
    head = args.head_exe.resolve(strict=True)
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    project = run_dir / "native-fixture"
    cache_root = run_dir / "cache"
    embeddings = cache_root / "embeddings"
    query_cache = embeddings / "query-vectors"
    config = run_dir / "empty-config.toml"
    report_path = run_dir / "benchmark-results.json"
    prepare_native_fixture(project)
    config.write_text('embedder = "minilm-l6-v2"\n', encoding="utf-8")
    model_copy = copy_model_cache(args.model_cache_root.resolve(strict=True), embeddings)
    revision, model_paths = active_model_files(embeddings)
    model_hashes_before = {name: sha256_file(path) for name, path in model_paths.items()}
    env = os.environ.copy()
    env.pop("HF_HOME", None)
    env.pop("VEX_EMBEDDER_SKIP_CHECK", None)
    env.pop("VEX_EMBEDDER", None)
    env.pop("RUST_LOG", None)
    env["HF_ENDPOINT"] = "http://127.0.0.1:9"
    env["HF_HUB_OFFLINE"] = "1"
    env["NO_PROXY"] = "127.0.0.1"
    report: dict[str, Any] = {
        "schema": "vex-query-cache-benchmark-v1",
        "started_utc": utc_now(),
        "method": (
            "One host; one isolated Rust/Python fixture; one shared semantic index; "
            "release baseline and head binaries in alternating order; at least five timed runs per side."
        ),
        "claim_under_test": (
            "A warm exact-query cache reduces repeated semantic CLI search time while preserving results."
        ),
        "machine": {
            "platform": platform.platform(),
            "python": sys.version,
            "logical_cpu_count": os.cpu_count(),
        },
        "inputs": {
            "baseline_exe": str(baseline),
            "baseline_sha256": sha256_file(baseline),
            "head_exe": str(head),
            "head_sha256": sha256_file(head),
            "model_copy": model_copy,
            "model_revision": revision,
            "embedder": "minilm-l6-v2",
            "embedder_selection": "explicit configuration; ambient VEX_EMBEDDER removed",
            "model_hashes": model_hashes_before,
            "query": QUERY,
            "query_utf8_sha256": sha256_bytes(QUERY.encode("utf-8")),
            "config": str(config),
            "config_sha256": sha256_file(config),
            "fixture_source_hashes": file_snapshot(project),
            "runs_per_side": args.runs,
        },
        "events": [],
        "controls": {},
        "validations": {},
        "summary": {},
        "complete": False,
    }
    recorder = Recorder(report_path, report, env, query_cache)
    recorder.save()
    common_baseline = [
        str(baseline),
        "--cache-dir",
        str(cache_root),
        "--config",
        str(config),
        "--format",
        "json",
    ]
    common_head = [
        str(head),
        "--cache-dir",
        str(cache_root),
        "--config",
        str(config),
        "--format",
        "json",
    ]
    search_tail = ["search", QUERY, "--semantic", "--limit", "10", "--no-stale-check"]
    cpu_before = system_cpu_times()
    try:
        for side, exe in (("baseline", baseline), ("head", head)):
            recorder.run("version", side, [str(exe), "--version"], project, timeout=args.timeout)
        index_row = recorder.run(
            "semantic_index",
            "baseline",
            common_baseline
            + ["index", "--path", str(project), "--semantic", "--no-gpu", "--jobs", "1"],
            project,
            timeout=args.timeout,
        )
        if not successful(index_row):
            raise RuntimeError("baseline semantic index failed")
        indexed_symbols = json.loads(index_row["stdout"])["results"]["symbols"]
        if indexed_symbols != 7:
            raise RuntimeError(f"expected seven fixture symbols, found {indexed_symbols}")
        if json.loads(index_row["stdout"])["results"].get("embeddings") is not True:
            raise RuntimeError("baseline fixture index has no semantic embeddings")
        report["inputs"]["indexed_symbols"] = indexed_symbols
        index_files_before = file_snapshot(cache_root, {"embeddings"})
        report["inputs"]["index_hashes"] = index_files_before
        baseline_warm = recorder.run(
            "search_warmup", "baseline", common_baseline + search_tail, project, timeout=args.timeout
        )
        head_first = recorder.run(
            "head_cache_first_use", "head", common_head + search_tail, project, timeout=args.timeout
        )
        if not successful(baseline_warm) or not successful(head_first):
            raise RuntimeError("search warmup failed")
        report["validations"]["warmup_results_equal"] = (
            baseline_warm["results_sha256"] == head_first["results_sha256"]
        )
        report["validations"]["warmup_stdout_equal"] = (
            baseline_warm["stdout_sha256"] == head_first["stdout_sha256"]
        )
        report["validations"]["warmup_stable_envelope_equal"] = (
            baseline_warm["stable_envelope_sha256"] == head_first["stable_envelope_sha256"]
        )
        first_cache = cache_snapshot(query_cache)
        report["validations"]["first_use_cache"] = first_cache
        query_suffix = ":" + sha256_bytes(QUERY.encode("utf-8"))
        report["validations"]["first_use_cache_valid"] = (
            len(first_cache) == 1
            and next(iter(first_cache.values())).get("count") == 1
            and next(iter(first_cache.values())).get("dim") == 384
            and next(iter(first_cache.values())).get("identity", "").endswith(query_suffix)
        )
        original_cache_path = query_cache_file(query_cache, QUERY)
        timed_rows: list[dict[str, Any]] = []
        for iteration in range(args.runs):
            order = ("baseline", "head") if iteration % 2 == 0 else ("head", "baseline")
            pair: dict[str, dict[str, Any]] = {}
            for side in order:
                row = recorder.run(
                    "timed_semantic_search",
                    side,
                    (common_baseline if side == "baseline" else common_head) + search_tail,
                    project,
                    iteration=iteration,
                    timeout=args.timeout,
                )
                timed_rows.append(row)
                pair[side] = row
            pair["head"]["paired_results_equal"] = (
                pair["head"]["results_sha256"] == pair["baseline"]["results_sha256"]
            )
            pair["head"]["paired_stdout_equal"] = (
                pair["head"]["stdout_sha256"] == pair["baseline"]["stdout_sha256"]
            )
            pair["head"]["paired_stable_envelope_equal"] = (
                pair["head"]["stable_envelope_sha256"] == pair["baseline"]["stable_envelope_sha256"]
            )
            recorder.save()
        version_rows: list[dict[str, Any]] = []
        for iteration in range(args.runs):
            order = ("baseline", "head") if iteration % 2 == 0 else ("head", "baseline")
            for side in order:
                version_rows.append(
                    recorder.run(
                        "process_start_control",
                        side,
                        [str(baseline if side == "baseline" else head), "--version"],
                        project,
                        iteration=iteration,
                        timeout=args.timeout,
                    )
                )
        artifact_hash_times = []
        for _ in range(args.runs):
            started = time.perf_counter_ns()
            observed = {name: sha256_file(path) for name, path in model_paths.items()}
            artifact_hash_times.append((time.perf_counter_ns() - started) / 1_000_000_000)
            if observed != model_hashes_before:
                raise RuntimeError("model artifacts changed during SHA control")
        report["controls"]["python_artifact_sha_probe"] = {
            "n": len(artifact_hash_times),
            "median_seconds": statistics.median(artifact_hash_times),
            "range_seconds": [min(artifact_hash_times), max(artifact_hash_times)],
            "times_seconds": artifact_hash_times,
            "scope": "Hashes the five guarded files once; the Rust cache-hit path validates identity twice.",
        }
        report["controls"]["process_start"] = {
            "baseline": summarize_times(version_rows, "baseline"),
            "head": summarize_times(version_rows, "head"),
        }
        report["validations"]["process_start_control_pass"] = (
            all(report["controls"]["process_start"][side]["n"] == args.runs for side in ("baseline", "head"))
            and all(row["exit_code"] == 0 for row in version_rows)
        )
        variant_tail = ["search", QUERY_VARIANT, "--semantic", "--limit", "10", "--no-stale-check"]
        variant_before = cache_snapshot(query_cache)
        variant_baseline = recorder.run(
            "exact_query_negative", "baseline", common_baseline + variant_tail, project, timeout=args.timeout
        )
        variant_head = recorder.run(
            "exact_query_negative", "head", common_head + variant_tail, project, timeout=args.timeout
        )
        variant_after = cache_snapshot(query_cache)
        report["validations"]["exact_query_negative"] = {
            "results_equal": variant_baseline["results_sha256"] == variant_head["results_sha256"],
            "stdout_equal": variant_baseline["stdout_sha256"] == variant_head["stdout_sha256"],
            "stable_envelope_equal": variant_baseline["stable_envelope_sha256"] == variant_head["stable_envelope_sha256"],
            "exits_zero": variant_baseline["exit_code"] == variant_head["exit_code"] == 0,
            "cache_files_before": len(variant_before),
            "cache_files_after": len(variant_after),
        }
        report["validations"]["exact_query_negative_pass"] = (
            report["validations"]["exact_query_negative"]["results_equal"]
            and report["validations"]["exact_query_negative"]["stable_envelope_equal"]
            and report["validations"]["exact_query_negative"]["exits_zero"]
            and len(variant_after) == len(variant_before) + 1
        )
        guarded_config = model_paths["tokenizer_config.json"]
        original_config = guarded_config.read_bytes()
        original_config_stat = guarded_config.stat()
        config_before = cache_snapshot(query_cache)
        try:
            in_string = escaped = False
            whitespace = None
            for position, byte in enumerate(original_config):
                if in_string:
                    if escaped:
                        escaped = False
                    elif byte == 92:
                        escaped = True
                    elif byte == 34:
                        in_string = False
                elif byte == 34:
                    in_string = True
                elif byte in b" \t\r\n":
                    whitespace = position
                    break
            if whitespace is None:
                raise ValueError("SHA-only control unavailable: config has no JSON whitespace outside strings")
            replacement = b" " if original_config[whitespace] == 9 else b"\t"
            changed_config = original_config[:whitespace] + replacement + original_config[whitespace + 1:]
            if len(changed_config) != len(original_config) or json.loads(changed_config) != json.loads(original_config):
                raise RuntimeError("SHA control must preserve JSON value and byte length")
            guarded_config.write_bytes(changed_config)
            os.utime(guarded_config, ns=(original_config_stat.st_atime_ns, original_config_stat.st_mtime_ns))
            report["validations"]["config_sha_only_change"] = (
                guarded_config.stat().st_size == original_config_stat.st_size
                and guarded_config.stat().st_mtime_ns == original_config_stat.st_mtime_ns
                and sha256_bytes(changed_config) != sha256_bytes(original_config)
            )
            changed_baseline = recorder.run(
                "config_sha_invalidation",
                "baseline",
                common_baseline + search_tail,
                project,
                timeout=args.timeout,
            )
            changed_head = recorder.run(
                "config_sha_invalidation", "head", common_head + search_tail, project, timeout=args.timeout
            )
            config_after = cache_snapshot(query_cache)
            report["validations"]["config_sha_invalidation"] = {
                "results_equal": changed_baseline["results_sha256"] == changed_head["results_sha256"],
                "stdout_equal": changed_baseline["stdout_sha256"] == changed_head["stdout_sha256"],
                "stable_envelope_equal": changed_baseline["stable_envelope_sha256"] == changed_head["stable_envelope_sha256"],
                "exits_zero": changed_baseline["exit_code"] == changed_head["exit_code"] == 0,
                "cache_files_before": len(config_before),
                "cache_files_after": len(config_after),
            }
            report["validations"]["config_sha_invalidation_pass"] = (
                report["validations"]["config_sha_invalidation"]["results_equal"]
                and report["validations"]["config_sha_invalidation"]["stable_envelope_equal"]
                and report["validations"]["config_sha_invalidation"]["exits_zero"]
                and len(config_after) == len(config_before) + 1
            )
        finally:
            guarded_config.write_bytes(original_config)
            os.utime(guarded_config, ns=(original_config_stat.st_atime_ns, original_config_stat.st_mtime_ns))
        valid_cache = original_cache_path.read_bytes()
        original_cache_path.write_bytes(b"BAD!" + valid_cache[4:])
        previous_log = recorder.env.get("RUST_LOG")
        recorder.env["RUST_LOG"] = "vex::embed::query_cache=debug"
        try:
            corrupt_fallback = recorder.run(
                "invalid_cache_fallback", "head", common_head + search_tail, project, timeout=args.timeout
            )
            report["validations"]["invalid_cache_fallback"] = {
                "exit_zero": corrupt_fallback["exit_code"] == 0,
                "results_equal_to_baseline": (
                    corrupt_fallback["results_sha256"] == baseline_warm["results_sha256"]
                ),
                "stdout_equal_to_baseline": (
                    corrupt_fallback["stdout_sha256"] == baseline_warm["stdout_sha256"]
                ),
                "stable_envelope_equal_to_baseline": (
                    corrupt_fallback["stable_envelope_sha256"] == baseline_warm["stable_envelope_sha256"]
                ),
                "fresh_factory_proven": "query vector cache: create fresh factory" in corrupt_fallback["stderr"],
            }
            report["validations"]["invalid_cache_fallback_pass"] = all(
                value for key, value in report["validations"]["invalid_cache_fallback"].items() if key != "stdout_equal_to_baseline"
            )
        finally:
            original_cache_path.write_bytes(valid_cache)
        restored_head = recorder.run(
            "restored_cache_hit", "head", common_head + search_tail, project, timeout=args.timeout
        )
        report["validations"]["restored_cache_results_equal"] = (
            successful(restored_head)
            and restored_head["results_sha256"] == baseline_warm["results_sha256"]
            and restored_head["stable_envelope_sha256"] == baseline_warm["stable_envelope_sha256"]
        )
        report["validations"]["cache_hit_bypasses_factory_proven"] = (
            "query vector cache: hit before factory" in restored_head["stderr"]
            and "query vector cache: create fresh factory" not in restored_head["stderr"]
        )
        if previous_log is None:
            recorder.env.pop("RUST_LOG", None)
        else:
            recorder.env["RUST_LOG"] = previous_log
        report["summary"]["timed_semantic_search"] = {
            "baseline": summarize_times(timed_rows, "baseline"),
            "head": summarize_times(timed_rows, "head"),
        }
        baseline_summary = report["summary"]["timed_semantic_search"]["baseline"]
        head_summary = report["summary"]["timed_semantic_search"]["head"]
        if baseline_summary["median_seconds"] and head_summary["median_seconds"]:
            report["summary"]["timed_semantic_search"]["median_ratio_baseline_over_head"] = (
                baseline_summary["median_seconds"] / head_summary["median_seconds"]
            )
            report["summary"]["timed_semantic_search"]["median_reduction_percent"] = 100 * (
                1 - head_summary["median_seconds"] / baseline_summary["median_seconds"]
            )
        paired_head = [row for row in timed_rows if row["side"] == "head"]
        report["validations"]["timed_results_equal"] = all(
            row.get("paired_results_equal") for row in paired_head
        )
        report["validations"]["timed_stdout_equal"] = all(
            row.get("paired_stdout_equal") for row in paired_head
        )
        report["validations"]["timed_stable_envelope_equal"] = all(
            row.get("paired_stable_envelope_equal") for row in paired_head
        )
        report["validations"]["timed_exits_zero"] = all(row["exit_code"] == 0 for row in timed_rows)
        report["validations"]["warm_cache_file_unchanged"] = all(
            row["cache_before"] == row["cache_after"] for row in paired_head
        )
        report["validations"]["source_unchanged"] = (
            report["inputs"]["fixture_source_hashes"] == file_snapshot(project)
        )
        report["validations"]["config_unchanged"] = (
            report["inputs"]["config_sha256"] == sha256_file(config)
        )
        report["validations"]["model_artifacts_restored"] = (
            model_hashes_before == {name: sha256_file(path) for name, path in model_paths.items()}
        )
        report["validations"]["index_unchanged"] = (
            index_files_before == file_snapshot(cache_root, {"embeddings"})
        )
        validity_keys = (
            "warmup_results_equal",
            "warmup_stable_envelope_equal",
            "first_use_cache_valid",
            "timed_results_equal",
            "timed_stable_envelope_equal",
            "timed_exits_zero",
            "warm_cache_file_unchanged",
            "exact_query_negative_pass",
            "config_sha_invalidation_pass",
            "config_sha_only_change",
            "process_start_control_pass",
            "invalid_cache_fallback_pass",
            "source_unchanged",
            "config_unchanged",
            "model_artifacts_restored",
            "index_unchanged",
            "restored_cache_results_equal",
            "cache_hit_bypasses_factory_proven",
        )
        invalid = [key for key in validity_keys if not report["validations"].get(key)]
        report["summary"]["validity_failures"] = invalid
        report["summary"]["number_story"] = (
            "inconclusive" if invalid else "measurement complete; limiter review required before a speed claim"
        )
        report["complete"] = True
        return 0 if not invalid else 2
    except Exception as failure:
        report["fatal_error"] = f"{type(failure).__name__}: {failure}"
        report["summary"]["number_story"] = "inconclusive"
        return 1
    finally:
        report["finished_utc"] = utc_now()
        report["machine"]["system_cpu_busy_percent"] = system_cpu_busy_percent(
            cpu_before, system_cpu_times()
        )
        recorder.save()


if __name__ == "__main__":
    raise SystemExit(main())
