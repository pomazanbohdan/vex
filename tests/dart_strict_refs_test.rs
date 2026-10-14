use assert_cmd::Command;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

fn command(root: &Path) -> Command {
    let mut cmd = Command::cargo_bin("vex").unwrap();
    cmd.current_dir(root)
        .env("VEX_CACHE_DIR", root.join(".vex-test-cache"));
    cmd
}

fn project() -> TempDir {
    let root = TempDir::new().unwrap();
    std::fs::write(root.path().join(".vex.toml"), "local_cache = true\n").unwrap();
    std::fs::write(
        root.path().join("lib.dart"),
        "void target() {}\nvoid caller() { target(); }\n",
    )
    .unwrap();
    command(root.path()).arg("index").assert().success();
    root
}

fn sdk() -> Option<PathBuf> {
    std::env::var_os("VEX_DART_SDK").map(PathBuf::from)
}

#[test]
fn strict_dart_resolves_with_real_analysis_server() {
    let Some(sdk) = sdk() else {
        eprintln!("SKIP: set VEX_DART_SDK to run the real Dart Analysis Server test");
        return;
    };
    let root = project();
    command(root.path())
        .env("VEX_DART_SDK", sdk)
        .args(["usages", "target", "--strict"])
        .assert()
        .success()
        .stdout(predicates::str::contains("lib.dart:2"));
}

#[cfg(windows)]
#[test]
fn strict_dart_uses_userprofile_when_cache_environment_is_empty() {
    let Some(sdk) = sdk() else {
        eprintln!("SKIP: set VEX_DART_SDK to run the real Dart Analysis Server test");
        return;
    };
    let root = project();
    command(root.path())
        .env("VEX_DART_SDK", sdk)
        .env("LOCALAPPDATA", "")
        .env("XDG_CACHE_HOME", "")
        .env_remove("HOME")
        .args(["usages", "target", "--strict"])
        .assert()
        .success()
        .stdout(predicates::str::contains("lib.dart:2"));
    assert!(!root.path().join("vex/dart-analysis").exists());
}

#[test]
fn missing_dart_sdk_is_explicit_and_ordinary_search_still_works() {
    let root = project();
    let unavailable = root.path().join("missing-sdk");
    command(root.path())
        .env("VEX_DART_SDK", &unavailable)
        .args(["usages", "target", "--strict"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("Dart strict backend unavailable"));
    command(root.path())
        .env("VEX_DART_SDK", unavailable)
        .args(["search", "target"])
        .assert()
        .success()
        .stdout(predicates::str::contains("target"));
}

#[test]
fn absent_dart_definition_returns_empty_without_an_sdk() {
    let root = project();
    let output = command(root.path())
        .env("VEX_DART_SDK", root.path().join("missing-sdk"))
        .args(["usages", "AbsentDefinition", "--strict", "--format", "json"])
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(1),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let data: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(data["results"].as_array().unwrap().is_empty());
    assert!(data["_meta"].get("vex.dev/reference_backend").is_none());
}

#[test]
fn absent_dart_definition_still_rejects_stale_source_without_an_sdk() {
    let root = project();
    std::fs::write(root.path().join("lib.dart"), "void AbsentDefinition() {}\n").unwrap();
    command(root.path())
        .env("VEX_DART_SDK", root.path().join("missing-sdk"))
        .args(["usages", "AbsentDefinition", "--strict", "--no-stale-check"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("source hashes are stale"));
}

#[test]
fn impact_shares_real_strict_backend_and_reports_provenance() {
    let Some(sdk) = sdk() else {
        return;
    };
    let root = project();
    let result = command(root.path())
        .env("VEX_DART_SDK", sdk)
        .args(["impact", "target", "--format", "json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: serde_json::Value = serde_json::from_slice(&result).unwrap();
    let channel = &json["results"]["channels"]["strict_refs"];
    assert_eq!(channel["available"], true);
    assert_eq!(channel["count"], 1);
    assert_eq!(channel["backend"]["kind"], "dart_analysis_server");
    assert_eq!(
        channel["backend"]["freshness"],
        "live_source_hashes_verified_before_and_after"
    );
    assert_eq!(channel["sample"][0]["line"], 2);
}

#[test]
fn strict_rejects_stale_source_even_with_no_stale_check() {
    let Some(sdk) = sdk() else {
        return;
    };
    let root = project();
    std::fs::write(
        root.path().join("lib.dart"),
        "\nvoid target() {}\nvoid caller() { target(); }\n",
    )
    .unwrap();
    command(root.path())
        .env("VEX_DART_SDK", sdk)
        .args(["usages", "target", "--strict", "--no-stale-check"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("source hashes are stale"));
}

#[test]
fn strict_timeout_is_bounded_and_next_query_recovers() {
    let Some(sdk) = sdk() else {
        return;
    };
    let root = project();
    let start = std::time::Instant::now();
    command(root.path())
        .env("VEX_DART_SDK", &sdk)
        .env("VEX_DART_TIMEOUT_MS", "1")
        .args(["usages", "target", "--strict"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("timeout"));
    assert!(start.elapsed() < std::time::Duration::from_secs(10));
    command(root.path())
        .env("VEX_DART_SDK", sdk)
        .env_remove("VEX_DART_TIMEOUT_MS")
        .args(["usages", "target", "--strict"])
        .assert()
        .success();
}

#[test]
fn strict_real_backend_preserves_scope_and_constructor_targets() {
    let Some(sdk) = sdk() else {
        return;
    };
    let root = project();
    std::fs::write(root.path().join("lib.dart"), "class Service {\n  Service();\n  Service.named();\n}\nvoid caller() { Service(); }\nvoid other() { Service.named(); }\n").unwrap();
    command(root.path()).arg("index").assert().success();
    let result = command(root.path())
        .env("VEX_DART_SDK", &sdk)
        .args(["usages", "named", "--strict", "--format", "json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: serde_json::Value = serde_json::from_slice(&result).unwrap();
    assert_eq!(json["results"].as_array().unwrap().len(), 1);
    assert_eq!(json["results"][0]["line"], 6);
    assert_eq!(
        json["_meta"]["vex.dev/reference_backend"]["potential_references"],
        false
    );
    command(root.path())
        .env("VEX_DART_SDK", sdk)
        .args(["usages", "named", "--strict", "--exclude", "lib.dart"])
        .assert()
        .code(1)
        .stdout(predicates::str::contains("No usages found"));
}

#[test]
fn strict_fvmrc_sdk_autodiscovery_requires_no_override() {
    let Some(sdk) = sdk() else {
        return;
    };
    let Some(version_root) = sdk.ancestors().find(|path| {
        path.parent()
            .is_some_and(|parent| parent.ends_with("fvm/versions"))
    }) else {
        return;
    };
    let version = version_root.file_name().unwrap().to_str().unwrap();
    let root = project();
    std::fs::write(
        root.path().join(".fvmrc"),
        serde_json::json!({"flutter": version}).to_string(),
    )
    .unwrap();
    command(root.path())
        .env_remove("VEX_DART_SDK")
        .args(["usages", "target", "--strict"])
        .assert()
        .success()
        .stdout(predicates::str::contains("lib.dart:2"));
}

#[test]
fn strict_cli_unions_resolved_definitions_without_dynamic_or_local_shadow_noise() {
    let Some(sdk) = sdk() else {
        return;
    };
    let root = project();
    std::fs::write(
        root.path().join("a.dart"),
        "void helper() {}\nclass A { void run() {} }\n",
    )
    .unwrap();
    std::fs::write(
        root.path().join("b.dart"),
        "void helper() {}\nclass B { void run() {} }\n",
    )
    .unwrap();
    std::fs::write(root.path().join("lib.dart"), "import 'a.dart' as a;\nimport 'b.dart' as b;\nvoid caller() {\n  a.helper();\n  b.helper();\n  a.A().run();\n  b.B().run();\n  final helper = () {}; helper();\n  dynamic receiver = a.A(); receiver.run();\n  final emoji = '🦊'; a.helper();\n}\n").unwrap();
    command(root.path()).arg("index").assert().success();
    for (name, expected) in [("helper", vec![4, 5, 10]), ("run", vec![6, 7])] {
        let output = command(root.path())
            .env("VEX_DART_SDK", &sdk)
            .args(["usages", name, "--strict", "--format", "json"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let json: serde_json::Value = serde_json::from_slice(&output).unwrap();
        let actual: Vec<_> = json["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| {
                assert_eq!(entry["path"], "lib.dart");
                entry["line"].as_u64().unwrap()
            })
            .collect();
        assert_eq!(actual, expected, "wrong refs for {name}");
    }
}

#[test]
fn strict_same_line_same_name_methods_preserve_all_resolved_identities() {
    let Some(sdk) = sdk() else {
        return;
    };
    let root = project();
    std::fs::write(root.path().join("lib.dart"), "class A { void run() {} } class B { void run() {} }\nvoid caller() {\n  A().run();\n  B().run();\n  dynamic d = A(); d.run();\n}\n").unwrap();
    command(root.path()).arg("index").assert().success();
    let output = command(root.path())
        .env("VEX_DART_SDK", sdk)
        .args(["usages", "run", "--strict", "--format", "json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: serde_json::Value = serde_json::from_slice(&output).unwrap();
    let lines: Vec<_> = json["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["line"].as_u64().unwrap())
        .collect();
    assert_eq!(lines, vec![3, 4]);
}

#[test]
fn strict_dart_does_not_merge_name_only_native_edges() {
    let Some(sdk) = sdk() else {
        return;
    };
    let root = project();
    std::fs::write(
        root.path().join("unrelated.py"),
        "def caller(obj):\n    obj.target()\n",
    )
    .unwrap();
    command(root.path()).arg("index").assert().success();
    let output = command(root.path())
        .env("VEX_DART_SDK", sdk)
        .args(["usages", "target", "--strict", "--format", "json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: serde_json::Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(
        json["results"],
        serde_json::json!([{"path":"lib.dart", "line":2}])
    );
}

#[test]
fn optional_dart_sdk_does_not_block_native_strict_in_mixed_project() {
    let root = project();
    std::fs::write(
        root.path().join("native.rs"),
        "fn native_helper() {}\nfn caller() {\n  native_helper();\n}\n",
    )
    .unwrap();
    command(root.path()).arg("index").assert().success();
    command(root.path())
        .env("VEX_DART_SDK", root.path().join("missing-sdk"))
        .args(["usages", "native_helper", "--strict"])
        .assert()
        .success()
        .stdout(predicates::str::contains("native.rs:3"));
}

#[test]
fn strict_full_name_excludes_camel_case_subtoken_declarations() {
    let Some(sdk) = sdk() else {
        return;
    };
    let root = project();
    std::fs::write(root.path().join("lib.dart"), "class Service {}\nclass DerivedService {}\nvoid caller() {\n  Service();\n  DerivedService();\n}\n").unwrap();
    command(root.path()).arg("index").assert().success();
    for name in ["Service", "service", "SERVICE"] {
        let output = command(root.path())
            .env("VEX_DART_SDK", &sdk)
            .args(["usages", name, "--strict", "--format", "json"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let json: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(
            json["results"],
            serde_json::json!([{"path":"lib.dart","line":4}]),
            "wrong declaration targets for {name}"
        );
    }
    let output = command(root.path())
        .env("VEX_DART_SDK", sdk)
        .args(["usages", "DerivedService", "--strict", "--format", "json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: serde_json::Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(
        json["results"],
        serde_json::json!([{"path":"lib.dart","line":5}])
    );
}

#[test]
fn dart_subtoken_does_not_require_sdk_for_exact_native_declaration() {
    let root = project();
    std::fs::write(
        root.path().join("lib.dart"),
        "class DerivedService {}\nvoid caller() { DerivedService(); }\n",
    )
    .unwrap();
    std::fs::write(
        root.path().join("native.rs"),
        "struct Service;\nfn caller() {\n  let _ = Service;\n}\n",
    )
    .unwrap();
    command(root.path()).arg("index").assert().success();
    command(root.path())
        .env("VEX_DART_SDK", root.path().join("missing-sdk"))
        .args(["usages", "Service", "--strict"])
        .assert()
        .success()
        .stdout(predicates::str::contains("native.rs:3"));
}

#[test]
fn strict_dart_target_excludes_native_camel_case_subtoken_edges() {
    let Some(sdk) = sdk() else {
        return;
    };
    let root = project();
    std::fs::write(
        root.path().join("lib.dart"),
        "class Service {}\nvoid caller() { Service(); }\n",
    )
    .unwrap();
    std::fs::write(
        root.path().join("native.rs"),
        "struct DerivedService;\nfn caller() {\n  let _ = DerivedService;\n}\n",
    )
    .unwrap();
    command(root.path()).arg("index").assert().success();
    let output = command(root.path())
        .env("VEX_DART_SDK", sdk)
        .args(["usages", "Service", "--strict", "--format", "json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: serde_json::Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(
        json["results"],
        serde_json::json!([{"path":"lib.dart","line":2}])
    );
}
