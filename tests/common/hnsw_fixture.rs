use std::path::Path;
use vex::index::symbols::{ParsedFile, ParsedSymbol, SymbolKind};

pub fn write_index(dir: &Path, vectors: &[Vec<f32>]) {
    let parsed = vec![ParsedFile {
        path: "fixture.rs".into(),
        symbols: (0..vectors.len())
            .map(|i| ParsedSymbol {
                name: format!("symbol_{i}"),
                kind: SymbolKind::Function,
                line: i + 1,
                signature: None,
                doc: None,
                body_tokens: None,
            })
            .collect(),
        refs: vec![],
        call_edges: vec![],
        bound_refs: vec![],
        skeletons: vec![],
        cpp_includes: vec![],
        trigram_bloom: None,
        hierarchy_captures: vec![],
    }];
    vex::store::writer::write_index_full(
        &parsed,
        vectors,
        vectors[0].len() as u32,
        &dir.join("index.vex"),
    )
    .unwrap();
}
