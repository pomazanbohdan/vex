use std::collections::HashMap;

use super::search_with_vector;
use crate::embed::cache::{context_hash, EmbedCache};
use crate::embed::{build_context, MINILM_CHAR_BUDGET, MINILM_DIM, MINILM_ID};
use crate::index::pipeline::{self, IndexOptions};
use crate::parse::{language::Language, parse_file};
use crate::search::hash_index;
use crate::store::reader::IndexReader;

#[test]
fn real_pipeline_preserves_cached_hash_vector_and_symbol_order() {
    if std::env::var_os("VEX_HNSW_PIPELINE_ORDER_CHILD").is_none() {
        let test_name = std::thread::current().name().unwrap().to_owned();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &test_name, "--nocapture"])
            .env("VEX_HNSW_PIPELINE_ORDER_CHILD", "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "isolated child failed: {stdout}{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            stdout.contains("running 1 test"),
            "child did not run exactly one test: {stdout}"
        );
        assert!(
            stdout.contains("1 passed; 0 failed"),
            "child did not pass the target test: {stdout}"
        );
        return;
    }
    let tmp = tempfile::TempDir::new().unwrap();
    let root = tmp.path();
    let cache_dir = root.join(".vex_cache");
    std::fs::create_dir(&cache_dir).unwrap();
    // Keep the process-wide override in an isolated one-test child process.
    crate::util::config::set_cache_override(cache_dir.clone(), true);
    std::fs::write(root.join(".vex.toml"), "local_cache = true\n").unwrap();

    let sources = [
        ("src/same.rs", "pub fn shared_needle() {}\n"),
        ("z.rs", "pub fn zebra() {}\npub fn apple() {}\n"),
        ("lib/same.rs", "pub fn shared_needle() {}\n"),
    ];
    let mut expected = HashMap::new();
    let mut cache = EmbedCache::empty(MINILM_ID, MINILM_DIM);
    for (path, source) in sources {
        let file = root.join(path);
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(file, source).unwrap();
        let parsed = parse_file(path, source, Language::Rust).unwrap();
        for symbol in parsed.symbols {
            let context = build_context(
                symbol.kind.as_str(),
                &symbol.name,
                path,
                symbol.signature.as_deref(),
                symbol.doc.as_deref(),
                symbol.body_tokens.as_deref(),
                MINILM_CHAR_BUDGET,
            );
            let hash = context_hash(MINILM_ID, &context);
            let mut vector = vec![0.0; MINILM_DIM as usize];
            let axis = match symbol.name.as_str() {
                "shared_needle" => 0,
                "zebra" => 1,
                "apple" => 2,
                other => panic!("unexpected symbol {other}"),
            };
            vector[axis] = 1.0;
            assert!(cache.insert(hash, vector.clone()));
            expected.insert((path.to_string(), symbol.name, symbol.line), (hash, vector));
        }
    }
    assert_eq!(cache.len(), 3, "two files must share one embedding context");
    cache
        .save(&crate::util::config::embed_cache_path(root, MINILM_ID))
        .unwrap();

    let options = IndexOptions {
        with_embeddings: true,
        with_call_graph: false,
        with_bm25: false,
        with_pattern_index: false,
        with_clusters: false,
        ..Default::default()
    };
    let (count, rebuilt) = pipeline::run(root, options, MINILM_ID, &[]).unwrap();
    assert_eq!(count, 4);
    assert!(rebuilt);
    let reader = IndexReader::open(&cache_dir.join("index.vex")).unwrap();
    let hashes = hash_index::load(&cache_dir.join("index.hashes")).unwrap();
    assert_eq!(reader.symbol_count(), expected.len());
    assert_eq!(hashes.len(), expected.len());
    let sidecar = std::fs::read(cache_dir.join("index.hashes")).unwrap();
    assert_eq!(&sidecar[4..8], &2_u32.to_le_bytes());
    let mut aliases = Vec::new();
    for (idx, &hash) in hashes.iter().enumerate() {
        let record = reader.symbol(idx).unwrap();
        let identity = (
            reader.read_string(record.file_offset).to_string(),
            reader.read_string(record.name_offset).to_string(),
            record.line as usize,
        );
        let (expected_hash, expected_vector) = expected
            .remove(&identity)
            .expect("persisted symbol identity");
        assert_eq!(hash, expected_hash, "hash order at sym_idx {idx}");
        assert_eq!(
            reader.vector(record.vector_index).unwrap(),
            expected_vector.as_slice(),
            "vector order at sym_idx {idx}"
        );
        if identity.1 == "shared_needle" {
            aliases.push(identity.0);
        }
    }
    assert!(expected.is_empty());
    assert_eq!(aliases.len(), 2);
    let results = search_with_vector(
        &reader,
        {
            let mut query = vec![0.0; MINILM_DIM as usize];
            query[0] = 1.0;
            query
        },
        1,
        &cache_dir.join("index.hnsw"),
        true,
    )
    .unwrap();
    let mut paths: Vec<_> = results.iter().map(|result| result.path.clone()).collect();
    paths.sort();
    aliases.sort();
    assert_eq!(
        paths, aliases,
        "one HNSW key expands to both production symbol positions"
    );
    assert!(results.iter().all(|result| result.score > 0.99));
}
