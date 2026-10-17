use std::{collections::BTreeMap, fs, path::Path};

use tempfile::TempDir;
use vex::{
    embed::{build_context, cache::context_hash, embedder_char_budget, embedder_dim},
    index::{
        manifest::Manifest,
        pipeline::{self, IndexOptions},
    },
    search::hash_index,
    store::{body_tokens, reader::IndexReader, writer::write_index_full},
    util::config,
};

const MODEL: &str = "minilm-l6-v2";
const SYMBOLS: [&str; 3] = ["prefix_alpha", "prefix_beta", "prefix_gamma"];
const A_SOURCE: &str =
    "pub fn prefix_alpha() { let seed_alpha = 1; }\npub fn prefix_beta() { prefix_alpha(); }\n";

fn opts() -> IndexOptions {
    IndexOptions {
        with_embeddings: false,
        with_clusters: false,
        with_pattern_index: false,
        ..IndexOptions::default()
    }
}

fn install_vectors(root: &Path, stored_count: usize) -> Result<(), Box<dyn std::error::Error>> {
    let index_path = config::index_path(root);
    let reader = IndexReader::open(&index_path)?;
    let tokens = body_tokens::load(&config::body_tokens_path(root))?;
    let budget = embedder_char_budget(MODEL).ok_or("unknown fixture model context budget")?;
    let dim = embedder_dim(MODEL).ok_or("unknown fixture model dimension")? as usize;
    let mut by_path = BTreeMap::<String, Vec<_>>::new();
    let mut vectors_by_path = BTreeMap::<String, Vec<Vec<f32>>>::new();
    let mut hashes_by_path = BTreeMap::<String, Vec<u64>>::new();
    for i in 0..reader.symbol_count() {
        let r = reader.symbol(i).ok_or("missing symbol record")?;
        let kind = vex::index::symbols::SymbolKind::try_from(r.kind)
            .map_err(|kind| format!("invalid fixture symbol kind: {kind}"))?;
        let name = reader.read_string(r.name_offset).to_string();
        let path = reader.read_string(r.file_offset).to_string();
        let signature = reader.read_string(r.signature_offset).to_string();
        let body = tokens.get(i).and_then(|value| value.as_deref());
        let context = build_context(
            kind.as_str(),
            &name,
            &path,
            (!signature.is_empty()).then_some(signature.as_str()),
            None,
            body,
            budget,
        );
        let slot = SYMBOLS
            .iter()
            .position(|candidate| *candidate == name)
            .ok_or_else(|| format!("unexpected fixture symbol {name}"))?;
        let mut vector = vec![0.0_f32; dim];
        vector[slot] = 1.0;
        let hash = context_hash(MODEL, &context);
        by_path
            .entry(path.clone())
            .or_default()
            .push(vex::index::symbols::ParsedSymbol {
                name,
                kind,
                line: r.line as usize,
                signature: (!signature.is_empty()).then_some(signature),
                doc: None,
                body_tokens: tokens.get(i).cloned().unwrap_or_default(),
            });
        vectors_by_path
            .entry(path.clone())
            .or_default()
            .push(vector);
        hashes_by_path.entry(path).or_default().push(hash);
    }

    // write_index_full replaces the FST; carry its ordinary refs into the seed.
    let mut refs_by_path = BTreeMap::<String, Vec<vex::index::symbols::ParsedRef>>::new();
    let paths = reader.file_paths();
    if let Some(refs) = reader.ref_reader() {
        refs.for_each_by_prefix("", |name, entries| {
            for entry in entries {
                let path = paths
                    .get(entry.file_id as usize)
                    .expect("valid fixture file id");
                refs_by_path.entry(path.clone()).or_default().push(
                    vex::index::symbols::ParsedRef {
                        name: name.to_owned(),
                        line: entry.line as usize,
                        context: None,
                    },
                );
            }
        });
    }
    drop(reader);

    let files = by_path
        .into_iter()
        .map(|(path, symbols)| vex::index::symbols::ParsedFile {
            refs: refs_by_path.remove(&path).unwrap_or_default(),
            path,
            symbols,
            call_edges: vec![],
            bound_refs: vec![],
            skeletons: vec![],
            cpp_includes: vec![],
            trigram_bloom: None,
            hierarchy_captures: vec![],
        })
        .collect::<Vec<_>>();
    assert_eq!(
        files.iter().map(|file| file.symbols.len()).sum::<usize>(),
        3,
        "fixture parser should emit exactly three symbols"
    );

    let mut all_vectors = Vec::new();
    let mut all_hashes = Vec::new();
    for file in &files {
        all_vectors.extend(
            vectors_by_path
                .remove(&file.path)
                .ok_or("missing vectors by path")?,
        );
        all_hashes.extend(
            hashes_by_path
                .remove(&file.path)
                .ok_or("missing hashes by path")?,
        );
    }
    assert!(stored_count <= all_vectors.len());
    write_index_full(
        &files,
        &all_vectors[..stored_count],
        dim as u32,
        &index_path,
    )?;
    body_tokens::save(
        &config::body_tokens_path(root),
        &files
            .iter()
            .flat_map(|file| file.symbols.iter().map(|symbol| symbol.body_tokens.clone()))
            .collect::<Vec<_>>(),
    )?;

    let hash_path = config::hash_index_path(root);
    let hnsw_path = config::hnsw_path(root);
    if stored_count == 3 {
        hash_index::save(&hash_path, &all_hashes)?;
        pipeline::build_hnsw_at(&hnsw_path, &hash_path, &all_vectors, &all_hashes)?;
    } else {
        for path in [&hash_path, &hnsw_path] {
            if path.exists() {
                fs::remove_file(path)?;
            }
        }
    }
    let mut manifest = Manifest::load(&config::manifest_path(root))?;
    manifest.embedder_id = Some(MODEL.to_string());
    manifest.vectors_normalized = Some(true);
    manifest.save(&config::manifest_path(root))?;
    Ok(())
}

fn assert_state(root: &Path, expected_prefix: usize, expect_hnsw: bool) {
    let reader = IndexReader::open(&config::index_path(root)).unwrap();
    assert_eq!(reader.symbol_count(), 3);
    assert_eq!(reader.header().vector_dim, embedder_dim(MODEL).unwrap());
    let mut saw_missing = false;
    let mut prefix_len = 0;
    for i in 0..reader.symbol_count() {
        let record = reader.symbol(i).unwrap();
        let name = reader.read_string(record.name_offset);
        let slot = SYMBOLS.iter().position(|symbol| *symbol == name).unwrap();
        if let Some(vector) = reader.vector(record.vector_index) {
            assert!(!saw_missing, "stored vectors must remain a symbol prefix");
            prefix_len += 1;
            assert_eq!(vector.len(), embedder_dim(MODEL).unwrap() as usize);
            for (dimension, value) in vector.iter().enumerate() {
                assert_eq!(
                    value.to_bits(),
                    if dimension == slot {
                        1.0_f32.to_bits()
                    } else {
                        0.0_f32.to_bits()
                    },
                    "vector identity mismatch for {name}"
                );
            }
        } else {
            saw_missing = true;
        }
    }
    assert_eq!(prefix_len, expected_prefix);
    assert_eq!(
        config::hash_index_path(root).exists(),
        expect_hnsw,
        "hash-index sidecar presence"
    );
    assert_eq!(
        config::hnsw_path(root).exists(),
        expect_hnsw,
        "HNSW sidecar presence"
    );
}

fn references(root: &Path, name: &str) -> Vec<(String, u32)> {
    let reader = IndexReader::open(&config::index_path(root)).unwrap();
    let paths = reader.file_paths();
    let mut refs = reader
        .ref_reader()
        .unwrap()
        .find(name)
        .into_iter()
        .map(|entry| (paths[entry.file_id as usize].clone(), entry.line))
        .collect::<Vec<_>>();
    refs.sort_unstable();
    refs
}

fn change_z(root: &Path, revision: u8) {
    fs::write(
        root.join("z_change.rs"),
        format!(
            "pub fn prefix_gamma() {{ let changed_gamma_{revision} = {}; }}\n",
            30 + revision as u32
        ),
    )
    .unwrap();
}

fn update(root: &Path) -> (usize, usize, usize) {
    pipeline::update(root, opts(), MODEL, &[]).unwrap()
}

#[test]
fn ordinary_updates_preserve_partial_vector_prefix_refs_and_hnsw_state(
) -> Result<(), Box<dyn std::error::Error>> {
    let tmp = TempDir::new()?;
    let root = tmp.path().canonicalize()?;
    fs::write(root.join(".vex.toml"), "local_cache = true\n")?;
    fs::write(root.join("a_vector.rs"), A_SOURCE)?;
    fs::write(
        root.join("z_change.rs"),
        "pub fn prefix_gamma() { prefix_beta(); let seed_gamma = 3; }\n",
    )?;
    let cfg = config::load_config(&root)?;
    let cache = config::resolve_cache_root(None, &cfg);
    config::set_cache_override(cache.root, cache.skip_hash_subdir);
    pipeline::run(&root, opts(), MODEL, &[])?;

    install_vectors(&root, 3)?;
    assert_state(&root, 3, true);
    change_z(&root, 1);
    assert_eq!(update(&root), (3, 1, 0));
    assert_state(&root, 2, false);
    assert_eq!(
        references(&root, "prefix_alpha"),
        [("a_vector.rs".into(), 1), ("a_vector.rs".into(), 2)]
    );
    assert_eq!(
        references(&root, "prefix_beta"),
        [("a_vector.rs".into(), 2)]
    );

    change_z(&root, 2);
    assert_eq!(update(&root), (3, 1, 0));
    assert_state(&root, 2, false);
    assert_eq!(
        references(&root, "prefix_alpha"),
        [("a_vector.rs".into(), 1), ("a_vector.rs".into(), 2)]
    );
    assert_eq!(update(&root), (3, 0, 0));
    assert_state(&root, 2, false);

    // The one-vector seed ends inside a_vector.rs, between its two symbols.
    install_vectors(&root, 1)?;
    assert_state(&root, 1, false);
    change_z(&root, 3);
    assert_eq!(update(&root), (3, 1, 0));
    assert_state(&root, 1, false);
    assert_eq!(
        references(&root, "prefix_alpha"),
        [("a_vector.rs".into(), 1), ("a_vector.rs".into(), 2)]
    );
    change_z(&root, 4);
    assert_eq!(update(&root), (3, 1, 0));
    assert_state(&root, 1, false);
    assert_eq!(update(&root), (3, 0, 0));
    assert_state(&root, 1, false);
    Ok(())
}
