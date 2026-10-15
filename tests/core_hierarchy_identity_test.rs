use tempfile::TempDir;
use vex::index::symbols::{HierarchyCapture, ParsedFile, ParsedSymbol, SymbolKind};
use vex::store::format::EdgeKind;
use vex::store::reader::IndexReader;
use vex::store::writer::write_index_full;

fn symbol(name: &str, kind: SymbolKind, line: usize) -> ParsedSymbol {
    ParsedSymbol {
        name: name.to_string(),
        kind,
        line,
        signature: None,
        doc: None,
        body_tokens: None,
    }
}

fn file(
    path: &str,
    symbols: Vec<ParsedSymbol>,
    hierarchy_captures: Vec<HierarchyCapture>,
) -> ParsedFile {
    ParsedFile {
        path: path.to_string(),
        symbols,
        refs: Vec::new(),
        call_edges: Vec::new(),
        bound_refs: Vec::new(),
        skeletons: Vec::new(),
        cpp_includes: Vec::new(),
        trigram_bloom: None,
        hierarchy_captures,
    }
}

fn capture(child_name: &str, parent_name: &str, line: usize) -> HierarchyCapture {
    HierarchyCapture {
        child_name: child_name.to_string(),
        parent_name: parent_name.to_string(),
        kind: EdgeKind::Extends as u8,
        line: line as u32,
    }
}

fn write(files: &[ParsedFile], tmp: &TempDir) -> IndexReader {
    let path = tmp.path().join("index.vex");
    write_index_full(files, &[], 384, &path).expect("write index");
    IndexReader::open(&path).expect("open index")
}

fn symbol_idx(reader: &IndexReader, name: &str, kind: SymbolKind) -> u32 {
    for idx in 0..reader.symbol_count() {
        let Some(rec) = reader.symbol(idx) else {
            continue;
        };
        if reader.read_string(rec.name_offset) == name && rec.kind == kind as u8 {
            return idx as u32;
        }
    }
    panic!("symbol {name} ({kind:?}) not found");
}

#[test]
fn hierarchy_ignores_same_named_non_types() {
    let tmp = TempDir::new().unwrap();
    let reader = write(
        &[file(
            "a.rs",
            vec![
                symbol("Base", SymbolKind::Method, 1),
                symbol("Base", SymbolKind::Class, 2),
                symbol("Child", SymbolKind::Class, 3),
            ],
            vec![capture("Child", "Base", 3)],
        )],
        &tmp,
    );

    let base_idx = symbol_idx(&reader, "Base", SymbolKind::Class);
    let child_idx = symbol_idx(&reader, "Child", SymbolKind::Class);
    let edges = reader.find_hierarchy_edges_by_symbol(base_idx);
    assert_eq!(edges.len(), 1, "the single type candidate must resolve");
    assert_eq!(edges[0].from_sym_idx, child_idx);
    assert!(reader.find_unresolved_hierarchy_by_name("Base").is_empty());
}

#[test]
fn hierarchy_ambiguous_types_do_not_resolve_or_spill() {
    let tmp = TempDir::new().unwrap();
    let reader = write(
        &[
            file("a.rs", vec![symbol("Base", SymbolKind::Class, 1)], vec![]),
            file("b.rs", vec![symbol("Base", SymbolKind::Struct, 1)], vec![]),
            file(
                "c.rs",
                vec![symbol("Child", SymbolKind::Class, 2)],
                vec![capture("Child", "Base", 2)],
            ),
        ],
        &tmp,
    );

    assert!(
        !reader.has_hierarchy_edges(),
        "ambiguous types must not resolve"
    );
    assert!(
        reader.find_unresolved_hierarchy_by_name("Base").is_empty(),
        "ambiguous types must not spill as external names"
    );
}

#[test]
fn hierarchy_zero_candidates_spill_verbatim() {
    let tmp = TempDir::new().unwrap();
    let reader = write(
        &[file(
            "a.rs",
            vec![symbol("Child", SymbolKind::Class, 2)],
            vec![capture("Child", "mymodule", 2)],
        )],
        &tmp,
    );

    let spilled = reader.find_unresolved_hierarchy_by_name("mymodule");
    assert_eq!(spilled.len(), 1, "zero candidates must spill verbatim");
    assert_eq!(spilled[0].edge_kind_bits(), EdgeKind::Extends as u8);
}
