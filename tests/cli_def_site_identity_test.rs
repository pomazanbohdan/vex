use std::path::Path;

use assert_cmd::Command;
use tempfile::TempDir;

fn vex_in(dir: &Path) -> Command {
    let mut cmd = Command::cargo_bin("vex").unwrap();
    cmd.current_dir(dir);
    cmd.env("VEX_CACHE_DIR", dir.join(".vex-test-cache"));
    cmd
}

fn index(dir: &Path) -> std::path::PathBuf {
    std::fs::write(dir.join(".vex.toml"), "local_cache = true\n").unwrap();
    let output = vex_in(dir)
        .args([
            "index",
            "--no-semantic",
            "--no-clusters",
            "--format",
            "json",
        ])
        .assert()
        .success();
    let json: serde_json::Value = serde_json::from_slice(&output.get_output().stdout).unwrap();
    std::path::PathBuf::from(json["results"]["index"].as_str().unwrap())
}

fn usages(dir: &Path, name: &str, extra: &[&str]) -> serde_json::Value {
    let output = vex_in(dir)
        .args(["usages", name, "--format", "json"])
        .args(extra)
        .output()
        .unwrap()
        .stdout;
    serde_json::from_slice(&output).expect("usages stdout is JSON")
}

#[test]
fn typescript_subtoken_definition_does_not_hide_same_line_exact_reference() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("src")).unwrap();
    std::fs::write(
        tmp.path().join("src/main.ts"),
        "class Service {}\nclass DerivedService extends Service {}\nfunction caller() { return new Service(); }\n",
    )
    .unwrap();
    let index_path = index(tmp.path());

    let expected = serde_json::json!([
        {"path":"src/main.ts","line":2},
        {"path":"src/main.ts","line":3}
    ]);
    assert_eq!(usages(tmp.path(), "Service", &[])["results"], expected);
    assert!(usages(tmp.path(), "service", &[])["results"]
        .as_array()
        .unwrap()
        .is_empty());
    let reader = vex::store::reader::IndexReader::open(&index_path).unwrap();
    let def_sites = vex::channel::build_def_sites(&reader, "service");
    assert_eq!(
        def_sites
            .get("src/main.ts")
            .map(|lines| lines.iter().copied().collect::<Vec<_>>()),
        Some(vec![1])
    );
    assert_eq!(
        usages(tmp.path(), "Service", &["--include-self"])["results"],
        serde_json::json!([
            {"path":"src/main.ts","line":1},
            {"path":"src/main.ts","line":2},
            {"path":"src/main.ts","line":3}
        ])
    );
}

#[test]
fn rust_subtoken_definition_does_not_hide_same_line_exact_reference() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("src")).unwrap();
    std::fs::write(
        tmp.path().join("src/main.rs"),
        "struct Service;\nstruct DerivedService { service: Service }\nfn use_service() { let _ = Service; }\n",
    )
    .unwrap();
    index(tmp.path());

    assert_eq!(
        usages(tmp.path(), "Service", &[])["results"],
        serde_json::json!([
            {"path":"src/main.rs","line":2},
            {"path":"src/main.rs","line":3}
        ])
    );
}
