use assert_cmd::Command;
use serde_json::{json, Value};
use std::path::Path;
use tempfile::TempDir;

fn command(root: &Path) -> Command {
    let mut command = Command::cargo_bin("vex").unwrap();
    command
        .current_dir(root)
        .env("VEX_CACHE_DIR", root.join(".vex-test-cache"));
    command
}

fn assert_references(source: &str, name: &str, lines: &[u32]) {
    let Some(sdk) = std::env::var_os("VEX_DART_SDK") else {
        eprintln!("SKIP: set VEX_DART_SDK for compiler accessor regression tests");
        return;
    };
    let root = TempDir::new().unwrap();
    std::fs::write(root.path().join(".vex.toml"), "local_cache = true\n").unwrap();
    std::fs::write(root.path().join("lib.dart"), source).unwrap();
    command(root.path()).arg("index").assert().success();
    let output = command(root.path())
        .env("VEX_DART_SDK", &sdk)
        .args(["usages", name, "--strict", "--format", "json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let result: Value = serde_json::from_slice(&output).unwrap();
    let expected: Vec<_> = lines
        .iter()
        .map(|line| json!({"path": "lib.dart", "line": line}))
        .collect();
    assert_eq!(result["results"], json!(expected), "wrong refs for {name}");
    assert_eq!(
        result["_meta"]["vex.dev/reference_backend"]["potential_references"],
        false
    );
    let runtime = &result["_meta"]["vex.dev/reference_backend"]["execution_runtime"];
    assert!(matches!(runtime["mode"].as_str(), Some("aot" | "jit")));
    for key in ["executable", "snapshot"] {
        assert!(
            Path::new(runtime[key].as_str().unwrap()).is_file(),
            "runtime {key} must be the launched file"
        );
    }
    command(root.path())
        .env("VEX_DART_SDK", sdk)
        .args(["usages", name, "--strict", "--exclude", "lib.dart"])
        .assert()
        .code(1)
        .stdout(predicates::str::contains("No usages found"));
}

#[test]
fn paired_accessors_union_reads_writes_and_compound_assignments() {
    assert_references(
        "class Box {\n  int _value = 0;\n  int get value => _value;\n  set value(int next) { _value = next; }\n}\nvoid caller() {\n  final box = Box();\n  print(box.value);\n  box.value = 1;\n  box.value += 2;\n  final emoji = '🦊'; print(box.value);\n  dynamic unknown = box; unknown.value = 3;\n  final value = 4; print(value);\n}\n",
        "value",
        &[8, 9, 10, 11],
    );
}

#[test]
fn getter_only_resolves_reads_without_local_or_dynamic_shadows() {
    assert_references(
        "class Box { int get readValue => 1; }\nvoid caller() {\n  final box = Box();\n  print(box.readValue);\n  final emoji = '🦊'; print(box.readValue);\n  dynamic unknown = box; print(unknown.readValue);\n  final readValue = 2; print(readValue);\n}\n",
        "readValue",
        &[4, 5],
    );
}

#[test]
fn setter_only_resolves_writes_without_local_or_dynamic_shadows() {
    assert_references(
        "class Box { set writeValue(int next) {} }\nvoid caller() {\n  final box = Box();\n  box.writeValue = 1;\n  final emoji = '🦊'; box.writeValue = 2;\n  dynamic unknown = box; unknown.writeValue = 3;\n  var writeValue = 4; writeValue = 5;\n}\n",
        "writeValue",
        &[4, 5],
    );
}

#[test]
fn top_level_accessors_exclude_local_shadows() {
    assert_references(
        "int _value = 0;\nint get globalValue => _value;\nset globalValue(int next) { _value = next; }\nvoid caller() {\n  print(globalValue);\n  globalValue = 1;\n  final emoji = '🦊'; globalValue += 2;\n}\nvoid shadow() { final globalValue = 5; print(globalValue); }\n",
        "globalValue",
        &[5, 6, 7],
    );
}

#[test]
fn same_line_same_name_accessors_preserve_distinct_owner_searches() {
    assert_references(
        "class A { int get value => 1; set value(int next) {} } class B { int get value => 2; set value(int next) {} }\nvoid caller() {\n  final a = A(); final b = B();\n  print(a.value);\n  a.value = 1;\n  print(b.value);\n  b.value = 2;\n  final emoji = '🦊'; print(a.value); print(b.value);\n  dynamic unknown = a; unknown.value = 3;\n  final value = 4; print(value);\n}\n",
        "value",
        &[4, 5, 6, 7, 8],
    );
}

#[test]
fn many_owner_accessor_unions_span_multiple_batches() {
    let mut source: String = (0..24)
        .map(|owner| {
            format!("class Box{owner} {{ int get value => {owner}; set value(int next) {{}} }}\n")
        })
        .collect();
    source.push_str("void caller() {\n");
    for owner in 0..24 {
        source.push_str(&format!("  Box{owner}().value = {owner};\n"));
    }
    source.push_str("}\n");
    assert_references(&source, "value", &(26..50).collect::<Vec<_>>());
}
