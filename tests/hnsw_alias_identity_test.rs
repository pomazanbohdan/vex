use std::path::Path;

use tempfile::TempDir;
use vex::embed::cache::context_hash;
use vex::embed::{build_context, Embedder, MINILM_CHAR_BUDGET, MINILM_DIM, MINILM_ID};
use vex::index::pipeline::build_hnsw_at;
use vex::index::symbols::{ParsedFile, ParsedSymbol, SymbolKind};
use vex::search::hash_index;
use vex::search::semantic::search_with_embedder;
use vex::store::reader::IndexReader;
use vex::store::writer::write_index_full;

struct FixtureEmbedder;

impl Embedder for FixtureEmbedder {
    fn id(&self) -> &'static str {
        MINILM_ID
    }

    fn dim(&self) -> u32 {
        MINILM_DIM
    }

    fn char_budget(&self) -> usize {
        MINILM_CHAR_BUDGET
    }

    fn embed(&mut self, _: &str) -> anyhow::Result<Vec<f32>> {
        Ok(one_hot(0))
    }

    fn embed_batch(&mut self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        texts.iter().map(|_| self.embed("")).collect()
    }
}

fn one_hot(slot: usize) -> Vec<f32> {
    let mut vector = vec![0.0; MINILM_DIM as usize];
    vector[slot] = 1.0;
    vector
}

fn symbol(name: &str, line: usize, body: &str) -> ParsedSymbol {
    ParsedSymbol {
        name: name.into(),
        kind: SymbolKind::Function,
        line,
        signature: Some(format!("fn {name}()")),
        doc: None,
        body_tokens: Some(body.into()),
    }
}

fn parsed(entries: &[(&str, usize, &str)]) -> Vec<ParsedFile> {
    vec![ParsedFile {
        path: "fixture.rs".into(),
        symbols: entries
            .iter()
            .map(|(name, line, body)| symbol(name, *line, body))
            .collect(),
        refs: vec![],
        call_edges: vec![],
        bound_refs: vec![],
        skeletons: vec![],
        cpp_includes: vec![],
        trigram_bloom: None,
        hierarchy_captures: vec![],
    }]
}

fn paths(tmp: &TempDir) -> (std::path::PathBuf, std::path::PathBuf) {
    (tmp.path().join("index.vex"), tmp.path().join("index.hnsw"))
}

fn write_index(path: &Path, files: &[ParsedFile], vectors: &[Vec<f32>]) {
    write_index_full(files, vectors, MINILM_DIM, path).expect("write_index_full");
}

fn write_graph(path: &Path, vectors: &[Vec<f32>], hashes: &[u64]) {
    build_hnsw_at(path, &path.with_file_name("index.hashes"), vectors, hashes)
        .expect("build pre-baked HNSW graph");
}

fn rewrite_hashes(path: &Path, hashes: &[u64]) {
    hash_index::save(&path.with_file_name("index.hashes"), hashes).expect("rewrite hash sidecar");
}

fn symbol_hashes(files: &[ParsedFile]) -> Vec<u64> {
    files
        .iter()
        .flat_map(|file| {
            file.symbols.iter().map(|symbol| {
                let context = build_context(
                    symbol.kind.as_str(),
                    &symbol.name,
                    &file.path,
                    symbol.signature.as_deref(),
                    symbol.doc.as_deref(),
                    symbol.body_tokens.as_deref(),
                    MINILM_CHAR_BUDGET,
                );
                context_hash(MINILM_ID, &context)
            })
        })
        .collect()
}

#[test]
fn hnsw_prebaked_aliases_keep_both_symbols() {
    let tmp = TempDir::new().unwrap();
    let (index_path, hnsw_path) = paths(&tmp);
    let files = parsed(&[
        ("same", 1, "shared"),
        ("same", 10, "shared"),
        ("other", 20, "other"),
    ]);
    let vectors = vec![one_hot(0), one_hot(0), one_hot(1)];
    write_index(&index_path, &files, &vectors);
    write_graph(&hnsw_path, &vectors, &symbol_hashes(&files));
    let reader = IndexReader::open(&index_path).unwrap();

    let results =
        search_with_embedder(&reader, &mut FixtureEmbedder, "query", 1, &hnsw_path, true).unwrap();
    assert_eq!(results.iter().map(|r| r.line).collect::<Vec<_>>(), [1, 10]);
}

#[test]
fn hnsw_rejects_foreign_graph() {
    let tmp = TempDir::new().unwrap();
    let (index_path, hnsw_path) = paths(&tmp);
    let files = parsed(&[("alpha", 1, "alpha"), ("beta", 2, "beta")]);
    let vectors = vec![one_hot(0), one_hot(1)];
    write_index(&index_path, &files, &vectors);
    // Same dimensions/cardinality, foreign graph keys, and the current
    // symbol sidecar. The graph must be rejected and search rebuilt.
    write_graph(&hnsw_path, &[one_hot(1), one_hot(0)], &[91, 92]);
    rewrite_hashes(&hnsw_path, &symbol_hashes(&files));
    let reader = IndexReader::open(&index_path).unwrap();

    let results =
        search_with_embedder(&reader, &mut FixtureEmbedder, "query", 2, &hnsw_path, true).unwrap();
    assert_eq!(
        results[0].line, 1,
        "foreign graph must not remap identities"
    );
}

#[test]
fn hnsw_reorder_does_not_remap_identity() {
    let tmp = TempDir::new().unwrap();
    let (index_path, hnsw_path) = paths(&tmp);
    let old_files = parsed(&[("alpha", 1, "alpha"), ("beta", 2, "beta")]);
    let files = parsed(&[("beta", 2, "beta"), ("alpha", 1, "alpha")]);
    let vectors = vec![one_hot(1), one_hot(0)];
    write_index(&index_path, &files, &vectors);
    // A stale graph/sidecar from the old symbol order.
    write_graph(
        &hnsw_path,
        &[one_hot(0), one_hot(1)],
        &symbol_hashes(&old_files),
    );
    rewrite_hashes(&hnsw_path, &symbol_hashes(&files));
    let reader = IndexReader::open(&index_path).unwrap();

    let results =
        search_with_embedder(&reader, &mut FixtureEmbedder, "query", 2, &hnsw_path, true).unwrap();
    assert_eq!(
        results[0].line, 1,
        "vector identity must follow reordered records"
    );
}

#[test]
fn hnsw_stale_sidecar_order_falls_back_after_reorder() {
    let tmp = TempDir::new().unwrap();
    let (index_path, hnsw_path) = paths(&tmp);
    let old_files = parsed(&[("alpha", 1, "alpha"), ("beta", 2, "beta")]);
    let current_files = parsed(&[("beta", 2, "beta"), ("alpha", 1, "alpha")]);
    let old_hashes = symbol_hashes(&old_files);
    let current_vectors = vec![one_hot(1), one_hot(0)];
    write_index(&index_path, &current_files, &current_vectors);
    write_graph(&hnsw_path, &[one_hot(0), one_hot(1)], &old_hashes);
    let reader = IndexReader::open(&index_path).unwrap();

    let results =
        search_with_embedder(&reader, &mut FixtureEmbedder, "query", 2, &hnsw_path, true).unwrap();
    assert_eq!(
        results[0].line, 1,
        "stale symbol order must use brute force"
    );
}

#[test]
fn hnsw_rejects_same_keys_with_wrong_vectors() {
    let tmp = TempDir::new().unwrap();
    let (index_path, hnsw_path) = paths(&tmp);
    let files = parsed(&[("alpha", 1, "alpha"), ("beta", 2, "beta")]);
    let vectors = vec![one_hot(0), one_hot(1)];
    let hashes = symbol_hashes(&files);
    write_index(&index_path, &files, &vectors);
    write_graph(&hnsw_path, &[one_hot(1), one_hot(0)], &hashes);
    let reader = IndexReader::open(&index_path).unwrap();

    let results =
        search_with_embedder(&reader, &mut FixtureEmbedder, "query", 2, &hnsw_path, true).unwrap();
    assert_eq!(
        results[0].line, 1,
        "same-key stale vectors must be rejected"
    );
}
