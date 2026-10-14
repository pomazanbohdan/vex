use vex::callgraph::extract_call_edges;
use vex::index::symbols::SymbolKind;
use vex::parse::extractor::extract_symbols_and_imports;
use vex::parse::language::Language;

fn dart() -> Language {
    Language::from_extension("dart").expect(".dart must be registered")
}

#[test]
fn dart_grammar_loads() {
    extract_symbols_and_imports("", dart()).expect("Dart grammar must load");
}

#[test]
fn dart_symbols_and_imports() {
    let source = include_str!("fixtures/sample.dart");
    let (symbols, imports) = extract_symbols_and_imports(source, dart()).unwrap();
    for name in ["Repository", "Logging", "Service"] {
        assert!(
            symbols
                .iter()
                .any(|s| s.name == name && s.kind == SymbolKind::Class),
            "missing class {name}: {symbols:?}"
        );
    }
    for name in ["helper", "run", "save", "refresh"] {
        assert!(
            symbols
                .iter()
                .any(|s| s.name == name
                    && matches!(s.kind, SymbolKind::Function | SymbolKind::Method)),
            "missing callable {name}: {symbols:?}"
        );
    }
    assert!(imports.iter().any(|r| r.name == "dart:async"));
}

#[test]
fn dart_calls_keep_function_ownership() {
    let source = include_str!("fixtures/sample.dart");
    let edges = extract_call_edges(source, dart());
    for (caller, callee) in [
        ("save", "helper"),
        ("save", "log"),
        ("refresh", "helper"),
        ("run", "save"),
        ("run", "refresh"),
    ] {
        assert!(
            edges
                .iter()
                .any(|(from, _, to, _)| from == caller && to == callee),
            "missing {caller}->{callee}: {edges:?}"
        );
    }
}

#[test]
fn dart_calls_exclude_bare_identifiers_and_comments() {
    let source = "void run() { final value = existing; print(value); /* ghost(); */ final text = 'fake()'; }";
    let edges = extract_call_edges(source, dart());
    assert!(edges
        .iter()
        .any(|(from, _, to, _)| from == "run" && to == "print"));
    for unwanted in ["existing", "value", "ghost", "fake"] {
        assert!(
            !edges.iter().any(|(_, _, to, _)| to == unwanted),
            "spurious {unwanted}: {edges:?}"
        );
    }
}

#[test]
fn dart_generic_and_conditional_calls() {
    let source = "void run(Client? client) { client?.fetch(); Future<void>.delayed(Duration.zero); List<int>.of([]); Box<int>(); new Crate<int>(); const Item<int>(); }";
    let edges = extract_call_edges(source, dart());
    for name in ["fetch", "delayed", "of", "Box", "Crate", "Item"] {
        assert!(
            edges
                .iter()
                .any(|(from, _, to, _)| from == "run" && to == name),
            "missing {name}: {edges:?}"
        );
    }
}

#[test]
fn dart_initializer_calls_belong_to_constructor() {
    let source = "int helper() => 1; class C { final int field; C() : field = helper(); }";
    let edges = extract_call_edges(source, dart());
    assert!(
        edges
            .iter()
            .any(|(from, _, to, _)| from == "C" && to == "helper"),
        "constructor owner lost: {edges:?}"
    );
}

#[test]
fn dart_enum_and_mixin_hierarchy() {
    use vex::parse::parse_file;
    use vex::store::format::EdgeKind;
    let source = "abstract class Contract {} mixin Logging {} mixin Worker implements Contract {} enum State with Logging implements Contract { ready; }";
    let parsed = parse_file("sample.dart", source, dart()).unwrap();
    for (child, base, kind) in [
        ("Worker", "Contract", EdgeKind::Implements),
        ("State", "Contract", EdgeKind::Implements),
        ("State", "Logging", EdgeKind::Uses),
    ] {
        assert!(
            parsed
                .hierarchy_captures
                .iter()
                .any(|c| c.child_name == child && c.parent_name == base && c.kind == kind as u8),
            "missing {child}->{base}: {:?}",
            parsed.hierarchy_captures
        );
    }
}

#[test]
fn dart_body_tokens_include_function_body() {
    let (symbols, _) =
        extract_symbols_and_imports(include_str!("fixtures/sample.dart"), dart()).unwrap();
    let refresh = symbols.iter().find(|s| s.name == "refresh").unwrap();
    assert!(
        refresh
            .body_tokens
            .as_deref()
            .unwrap_or("")
            .contains("helper"),
        "body tokens lost: {refresh:?}"
    );
}

#[test]
fn dart_type_aliases_do_not_capture_return_types() {
    let source =
        "typedef Items = List<int>; typedef Callback = void Function(); typedef int Legacy();";
    let (symbols, _) = extract_symbols_and_imports(source, dart()).unwrap();
    let aliases: Vec<_> = symbols
        .iter()
        .filter(|s| s.kind == SymbolKind::TypeAlias)
        .map(|s| s.name.as_str())
        .collect();
    assert_eq!(aliases, ["Items", "Callback", "Legacy"]);
}

#[test]
fn dart_hierarchy_keeps_relation_kinds() {
    use vex::parse::parse_file;
    use vex::store::format::EdgeKind;
    let source = "class Base {} abstract class Contract {} mixin Logging {} class Child extends Base with Logging implements Contract {}";
    let parsed = parse_file("sample.dart", source, dart()).unwrap();
    for (base, kind) in [
        ("Base", EdgeKind::Extends),
        ("Contract", EdgeKind::Implements),
        ("Logging", EdgeKind::Uses),
    ] {
        assert!(
            parsed
                .hierarchy_captures
                .iter()
                .any(|c| c.child_name == "Child" && c.parent_name == base && c.kind == kind as u8),
            "missing Child->{base}: {:?}",
            parsed.hierarchy_captures
        );
    }
}

#[test]
fn dart_ast_refs_keep_interpolation_and_drop_literal_text() {
    use vex::parse::extractor::extract_references_ast;
    let source = "void run() { final text = 'literalGhost ${actualValue}'; // commentGhost\n }";
    let refs = extract_references_ast(source, dart()).unwrap();
    assert!(refs.iter().any(|r| r.name == "actualValue"));
    assert!(!refs
        .iter()
        .any(|r| r.name == "literalGhost" || r.name == "commentGhost"));
}
