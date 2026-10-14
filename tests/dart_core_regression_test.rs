use std::path::Path;

use assert_cmd::Command;
use serde_json::Value;
use tempfile::TempDir;
use vex::index::symbols::SymbolKind;
use vex::parse::extractor::{extract_references, extract_references_ast};
use vex::parse::language::Language;
use vex::parse::parse_file;
use vex::store::{reader::IndexReader, writer::write_index};

fn vex_in(dir: &Path) -> Command {
    let mut cmd = Command::cargo_bin("vex").unwrap();
    cmd.current_dir(dir)
        .env("VEX_CACHE_DIR", dir.join(".cache"));
    cmd
}

fn json(dir: &Path, args: &[&str]) -> Value {
    let output = vex_in(dir)
        .args(args)
        .args(["--format", "json"])
        .output()
        .unwrap();
    assert!(
        matches!(output.status.code(), Some(0 | 1)),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn project(source: &str) -> TempDir {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("main.dart"), source).unwrap();
    vex_in(dir.path())
        .args(["index", "--no-semantic", "--no-clusters"])
        .assert()
        .success();
    dir
}

#[test]
fn dart_constructor_collision_preserves_transitive_subtypes() {
    let source = "abstract class Repository {}\nclass Service implements Repository {\n Service();\n}\nclass DerivedService extends Service {\n DerivedService();\n}\nvoid run() { Service(); DerivedService(); }\n";
    let dir = project(source);
    let subtypes = json(dir.path(), &["subtypes", "Repository"]);
    let results = subtypes["results"].as_array().unwrap();
    assert!(
        results
            .iter()
            .any(|r| r["name"] == "Service" && r["depth"] == 1),
        "{subtypes}"
    );
    assert!(
        results
            .iter()
            .any(|r| r["name"] == "DerivedService" && r["depth"] == 2),
        "{subtypes}"
    );
    let parsed = parse_file("main.dart", source, Language::Dart).unwrap();
    for name in ["Service", "DerivedService"] {
        assert!(parsed
            .symbols
            .iter()
            .any(|s| s.name == name && s.kind == SymbolKind::Method));
        assert!(parsed
            .call_edges
            .iter()
            .any(|edge| edge.callee_name == name));
    }
}

#[test]
fn dart_distinct_ambiguous_parent_types_remain_unresolved() {
    let files = [
        parse_file("a.dart", "class Service { Service(); }", Language::Dart).unwrap(),
        parse_file("b.dart", "class Service { Service(); }", Language::Dart).unwrap(),
        parse_file(
            "child.dart",
            "class Child extends Service {}",
            Language::Dart,
        )
        .unwrap(),
    ];
    let dir = TempDir::new().unwrap();
    let index = dir.path().join("index.vex");
    write_index(&files, &index).unwrap();
    let reader = IndexReader::open(&index).unwrap();
    assert!(!reader.has_hierarchy_edges());
    assert!(reader
        .find_unresolved_hierarchy_by_name("Service")
        .is_empty());
}

#[test]
fn dart_usages_keep_superclass_refs_on_other_declarations() {
    let dir = project("class Service {\n Service();\n}\nclass DerivedService extends Service {}\nvoid run() { Service(); }\n");
    let usages = json(dir.path(), &["usages", "Service"]);
    let results = usages["results"].as_array().unwrap();
    assert!(
        results.iter().any(|r| r["line"] == 4),
        "superclass ref pruned: {usages}"
    );
    assert!(
        !results.iter().any(|r| r["line"] == 1 || r["line"] == 2),
        "own definitions retained: {usages}"
    );
}

#[test]
fn dart_ast_refs_accept_lowercase_and_short_code_identifiers() {
    let source = "void save() {}\nint id = 0;\nclass C {}\nvoid run() { save(); print(id); C(); final text = 'literalghost ${id} $id'; /* commentghost */ }\n";
    let refs = extract_references_ast(source, Language::Dart).unwrap();
    for name in ["save", "id", "C"] {
        assert!(
            refs.iter().any(|r| r.name == name && r.line == 4),
            "missing {name}: {refs:?}"
        );
    }
    assert_eq!(
        refs.iter()
            .filter(|r| r.name == "id" && r.line == 4)
            .count(),
        3
    );
    for name in ["literalghost", "commentghost", "void", "final"] {
        assert!(
            !refs.iter().any(|r| r.name == name),
            "spurious {name}: {refs:?}"
        );
    }
    assert!(extract_references("lowercase id C // NamedGhost")
        .iter()
        .all(|r| r.name == "NamedGhost"));
}

#[test]
fn dart_ast_refs_keep_identifiers_that_are_other_language_keywords() {
    let source = "void open() {} void from() {} void type() {} void fn() {} void match() {} void Ok() {}\nvoid run() { open(); from(); type(); fn(); match(); Ok(); }\n";
    let refs = extract_references_ast(source, Language::Dart).unwrap();
    for name in ["open", "from", "type", "fn", "match", "Ok"] {
        assert!(
            refs.iter().any(|r| r.name == name && r.line == 2),
            "legal Dart code identifier rejected: {name}: {refs:?}"
        );
    }
    let dir = project(source);
    for name in ["open", "from", "type", "fn", "match", "Ok"] {
        let usages = json(dir.path(), &["usages", name]);
        assert_eq!(
            usages["results"].as_array().unwrap().len(),
            1,
            "{name}: {usages}"
        );
    }
}

#[test]
fn dart_test_path_classification_drives_usages_scope() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("main.dart"),
        "void save() {}\nvoid run() { save(); }\n",
    )
    .unwrap();
    for path in ["sample_test.dart", "integration_test/app.dart"] {
        let dest = dir.path().join(path);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(dest, "void checkSave() { save(); }\n").unwrap();
    }
    vex_in(dir.path())
        .args(["index", "--no-semantic", "--no-clusters"])
        .assert()
        .success();
    let all = json(dir.path(), &["usages", "save"]);
    assert_eq!(all["results"].as_array().unwrap().len(), 3, "{all}");
    let kept = json(dir.path(), &["usages", "save", "--exclude-tests"]);
    let results = kept["results"].as_array().unwrap();
    assert_eq!(results.len(), 1, "{kept}");
    assert_eq!(results[0]["path"], "main.dart");
}
