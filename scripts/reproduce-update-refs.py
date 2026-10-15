#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Reproduce loss of references from an unchanged file.")
    parser.add_argument("--vex", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expect", required=True, choices=["before", "after"])
    args = parser.parse_args()
    exe = args.vex.resolve(strict=True)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    project = run_dir / "project"
    project.mkdir()
    config = project / ".vex.toml"
    config.write_text("semantic = false\nauto_update = false\n", encoding="utf-8")
    (project / "target.rs").write_text("pub fn target_name() {}\n", encoding="utf-8")
    caller = project / "caller.rs"
    caller.write_text("fn caller() { target_name(); }\n", encoding="utf-8")
    edited = project / "edited.rs"
    edited.write_text("fn unrelated() {}\n", encoding="utf-8")
    caller_hash = hashlib.sha256(caller.read_bytes()).hexdigest()
    common = [str(exe), "--config", str(config), "--cache-dir", str(run_dir / "cache"), "--format", "json"]
    rows = []

    def run(label, argv):
        p = subprocess.run(common + argv, cwd=project, capture_output=True, text=True, encoding="utf-8", timeout=300)
        row = {"label": label, "argv": argv, "exit": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
        rows.append(row)
        row["data"] = json.loads(p.stdout)
        if not isinstance(row["data"], dict) or row["data"].get("protocol_version") != "v1" or "results" not in row["data"]:
            raise ValueError(f"invalid Vex JSON envelope: {label}")
        return row

    indexed = run("initial index", ["index", "--no-semantic", "--no-clusters", "--no-pattern-index"])
    before = run("references before unrelated edit", ["usages", "target_name"])
    edited.write_text("fn unrelated() { let changed = 1; }\n", encoding="utf-8")
    updated = run("incremental update", ["update", "--no-semantic"])
    after = run("references after unrelated edit", ["usages", "target_name"])
    contains_caller = lambda row: row["data"] is not None and any(x["path"] == "caller.rs" for x in row["data"]["results"])
    checks = {
        "index": indexed["exit"] == 0,
        "initial_reference": before["exit"] == 0 and before["data"]["results"] == [{"line": 1, "path": "caller.rs"}],
        "update": updated["exit"] == 0,
        "caller_source_unchanged": hashlib.sha256(caller.read_bytes()).hexdigest() == caller_hash,
        "expected_reference_state": contains_caller(after) == (args.expect == "after"),
    }
    if args.expect == "after":
        checks["references_preserved"] = after["exit"] == 0 and before["data"]["results"] == after["data"]["results"]
    else:
        checks["references_lost"] = after["exit"] == 1 and after["data"]["results"] == []
    report = {"exe": str(exe), "sha256": hashlib.sha256(exe.read_bytes()).hexdigest(), "expect": args.expect, "checks": checks, "commands": rows, "pass": all(checks.values())}
    (run_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "checks": checks, "report": str(run_dir / "result.json")}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
