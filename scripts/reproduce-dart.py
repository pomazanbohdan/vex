#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Compare Dart support with an upstream release binary.")
    parser.add_argument("--vex", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expect", required=True, choices=["before", "after"])
    parser.add_argument("--sdk", type=Path)
    args = parser.parse_args()
    exe = args.vex.resolve(strict=True)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    project = run_dir / "project"
    project.mkdir()
    config = project / ".vex.toml"
    config.write_text("semantic = false\nauto_update = false\n", encoding="utf-8")
    (project / "lib.dart").write_text(
        "abstract class Repository { void save(); }\n"
        "class Service extends Repository { void save() { helper(); } }\n"
        "void helper() {}\n"
        "void run() { Service().save(); }\n", encoding="utf-8"
    )
    (project / "sample_test.dart").write_text(
        "import 'lib.dart';\nvoid main() { helper(); }\n", encoding="utf-8"
    )
    (project / "native.rs").write_text("pub fn native_control() {}\nfn native_caller() { native_control(); }\n", encoding="utf-8")
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in project.iterdir()}
    env = dict(os.environ)
    sandbox_home = run_dir / "home"
    sandbox_home.mkdir()
    cache_base = run_dir / "platform-cache"
    cache_base.mkdir()
    env.update(HOME=str(sandbox_home), USERPROFILE=str(sandbox_home), LOCALAPPDATA=str(cache_base), XDG_CACHE_HOME=str(cache_base))
    if args.sdk:
        env["VEX_DART_SDK"] = str(args.sdk.resolve(strict=True))
    if args.expect == "after" and not env.get("VEX_DART_SDK"):
        raise ValueError("--sdk or VEX_DART_SDK is required for the after compiler check")
    common = [str(exe), "--config", str(config), "--cache-dir", str(run_dir / "cache"), "--format", "json"]
    rows = []

    def run(label, argv):
        p = subprocess.run(common + argv, cwd=project, env=env, capture_output=True, text=True, encoding="utf-8", timeout=300)
        row = {"label": label, "argv": argv, "exit": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
        rows.append(row)
        row["data"] = json.loads(p.stdout)
        if not isinstance(row["data"], dict) or row["data"].get("protocol_version") != "v1" or "results" not in row["data"]:
            raise ValueError(f"invalid Vex JSON envelope: {label}")
        return row

    indexed = run("index", ["index", "--no-semantic", "--no-clusters", "--no-pattern-index"])
    native = run("native control", ["check", "native_control"])
    symbols = run("Dart declarations", ["check", "Repository", "Service", "helper"])
    refs = run("Dart ordinary references", ["usages", "helper"])
    hierarchy = run("Dart hierarchy", ["implementations", "Repository"])
    tests = run("Dart main entrypoint", ["tests-for", "helper"])
    strict = run("Dart compiler references", ["usages", "helper", "--strict"])
    expected_symbols = [{"exists": args.expect == "after", "name": name} for name in ["Repository", "Service", "helper"]]
    expected_sites = [("lib.dart", 2), ("sample_test.dart", 2)]
    sites = lambda row: sorted((item["path"], item["line"]) for item in row["data"]["results"])
    checks = {"index": indexed["exit"] == 0, "native_control": native["exit"] == 0 and native["data"]["results"] == [{"exists": True, "name": "native_control"}]}
    if args.expect == "after":
        checks.update(
            declarations=symbols["exit"] == 0 and symbols["data"]["results"] == expected_symbols,
            references=refs["exit"] == 0 and sites(refs) == expected_sites,
            hierarchy=hierarchy["exit"] == 0 and [(x["name"], x["path"], x["line"]) for x in hierarchy["data"]["results"]] == [("Service", "lib.dart", 2)],
            main=tests["exit"] == 0 and [(x["name"], x["path"], x["line"]) for x in tests["data"]["results"]] == [("main", "sample_test.dart", 2)],
            compiler=strict["exit"] == 0 and sites(strict) == expected_sites
            and strict["data"]["_meta"].get("vex.dev/reference_backend", {}).get("kind") == "dart_analysis_server",
        )
    else:
        checks.update(
            declarations=symbols["exit"] == 0 and symbols["data"]["results"] == expected_symbols,
            references=refs["exit"] == 1 and refs["data"]["results"] == [],
            hierarchy=hierarchy["exit"] == 1 and hierarchy["data"]["results"] == [],
            main=tests["exit"] == 1 and tests["data"]["results"] == [],
            compiler=strict["exit"] == 1 and strict["data"]["results"] == [],
        )
    checks["source_unchanged"] = source_hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in project.iterdir()}
    report = {"exe": str(exe), "sha256": hashlib.sha256(exe.read_bytes()).hexdigest(), "expect": args.expect, "checks": checks, "commands": rows, "pass": all(checks.values())}
    (run_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "checks": checks, "report": str(run_dir / "result.json")}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
