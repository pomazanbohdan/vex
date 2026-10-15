use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use assert_cmd::Command;
use serde_json::Value;
use tempfile::TempDir;
use vex::index::symbols::ParsedFile;
use vex::parse::language::Language;
use vex::parse::parse_file;
use vex::store::reader::IndexReader;

const EMBEDDER: &str = "minilm-l6-v2";
const OTHER_EMBEDDER: &str = "jina-code";

fn vex_in(dir: &Path) -> Command {
    let mut command = Command::cargo_bin("vex").unwrap();
    command
        .current_dir(dir)
        .env("VEX_CACHE_DIR", dir.join(".vex-test-cache"))
        .env_remove("VEX_EMBEDDER")
        .env_remove("VEX_DEVICE")
        .env("HF_HUB_OFFLINE", "1");
    command
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

fn index_project(dir: &Path) -> PathBuf {
    let output = json(dir, &["index", "--no-semantic", "--no-clusters"]);
    PathBuf::from(output["results"]["index"].as_str().unwrap())
}

fn ordinary_refs(index: &Path, name: &str) -> Vec<(String, u32)> {
    let reader = IndexReader::open(index).unwrap();
    let paths = reader.file_paths();
    let file_ids: HashMap<&str, u32> = paths
        .iter()
        .enumerate()
        .map(|(idx, path)| (path.as_str(), idx as u32))
        .collect();
    let definition_sites: HashSet<(u32, u32)> = (0..reader.symbol_count())
        .filter_map(|idx| {
            let record = reader.symbol(idx)?;
            (reader.read_string(record.name_offset) == name).then(|| {
                let path = reader.read_string(record.file_offset);
                (file_ids[path], record.line)
            })
        })
        .collect();
    let mut refs = reader
        .ref_reader()
        .map(|refs| {
            refs.find(name)
                .into_iter()
                .filter(|entry| !definition_sites.contains(&(entry.file_id, entry.line)))
                .map(|entry| (paths[entry.file_id as usize].clone(), entry.line))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    refs.sort();
    refs
}

fn one_hot(dim: u32, slot: usize) -> Vec<f32> {
    let mut vector = vec![0.0; dim as usize];
    vector[slot] = 1.0;
    vector
}

fn seed_embed_cache(
    index: &Path,
    embedder_id: &str,
    files: &[(&str, &str)],
    slots: &[(&str, usize)],
) -> HashMap<String, Vec<f32>> {
    let dim = vex::embed::embedder_dim(embedder_id).unwrap();
    let budget = vex::embed::embedder_char_budget(embedder_id).unwrap();
    let slots: HashMap<&str, usize> = slots.iter().copied().collect();
    let mut cache = vex::embed::cache::EmbedCache::empty(embedder_id, dim);
    let mut expected = HashMap::new();
    for (path, source) in files {
        let parsed = parse_file(path, source, Language::Rust).unwrap();
        for symbol in parsed.symbols {
            let slot = slots[&*symbol.name];
            let vector = one_hot(dim, slot);
            let context = vex::embed::build_context(
                symbol.kind.as_str(),
                &symbol.name,
                path,
                symbol.signature.as_deref(),
                symbol.doc.as_deref(),
                symbol.body_tokens.as_deref(),
                budget,
            );
            let hash = vex::embed::cache::context_hash(embedder_id, &context);
            assert!(cache.insert(hash, vector.clone()));
            assert!(expected.insert(symbol.name, vector).is_none());
        }
    }
    assert_eq!(expected.len(), slots.len());
    cache
        .save(&index.with_file_name(format!("embed_cache_{embedder_id}.bin")))
        .unwrap();
    expected
}

fn assert_vectors_by_name(index: &Path, expected: &HashMap<String, Vec<f32>>) {
    let reader = IndexReader::open(index).unwrap();
    assert_eq!(reader.symbol_count(), expected.len());
    for idx in 0..reader.symbol_count() {
        let record = reader.symbol(idx).unwrap();
        let name = reader.read_string(record.name_offset);
        let expected_vector = expected.get(name).unwrap();
        assert_eq!(
            reader.vector(record.vector_index),
            Some(expected_vector.as_slice()),
            "wrong vector for {name}"
        );
    }
}

#[test]
fn incremental_updates_preserve_native_refs_and_occurrence_multiplicity() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join(".vex.toml"), "local_cache = true\n").unwrap();
    std::fs::write(dir.path().join("00_deleted.py"), "value = removed_name\n").unwrap();
    std::fs::write(
        dir.path().join("refs_only.py"),
        "pair = (target_name, target_name)\n",
    )
    .unwrap();
    std::fs::write(dir.path().join("target.rs"), "pub fn target_name() {}\n").unwrap();
    std::fs::write(dir.path().join("zz_changed.py"), "value = stale_name\n").unwrap();
    let index = index_project(dir.path());

    assert_eq!(
        ordinary_refs(&index, "target_name"),
        vec![("refs_only.py".into(), 1), ("refs_only.py".into(), 1)]
    );

    std::fs::remove_file(dir.path().join("00_deleted.py")).unwrap();
    std::fs::write(dir.path().join("zz_changed.py"), "value = fresh_name\n").unwrap();
    json(dir.path(), &["update"]);

    assert_eq!(
        ordinary_refs(&index, "target_name"),
        vec![("refs_only.py".into(), 1), ("refs_only.py".into(), 1)]
    );
    assert!(ordinary_refs(&index, "removed_name").is_empty());
    assert!(ordinary_refs(&index, "stale_name").is_empty());
    assert_eq!(
        ordinary_refs(&index, "fresh_name"),
        vec![("zz_changed.py".into(), 1)]
    );

    std::fs::write(
        dir.path().join("zz_changed.py"),
        "value = fresh_name\nsecond = 2\n",
    )
    .unwrap();
    json(dir.path(), &["update"]);
    assert_eq!(
        ordinary_refs(&index, "target_name"),
        vec![("refs_only.py".into(), 1), ("refs_only.py".into(), 1)]
    );
}

#[test]
fn old_revision_auto_update_repairs_poisoned_ordinary_refs_once() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join(".vex.toml"), "local_cache = true\n").unwrap();
    let source = "pub fn target_name() {}\nfn caller() { target_name(); }\n";
    std::fs::write(dir.path().join("lib.rs"), source).unwrap();
    let index = index_project(dir.path());

    let mut parsed = parse_file("lib.rs", source, Language::Rust).unwrap();
    parsed.refs.clear();
    vex::store::writer::write_index(&[parsed], &index).unwrap();
    let manifest_path = index.with_file_name("manifest.json");
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    manifest.as_object_mut().unwrap().remove("parse_revision");
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(ordinary_refs(&index, "target_name").is_empty());

    json(dir.path(), &["usages", "target_name", "--auto-update"]);
    assert_eq!(
        ordinary_refs(&index, "target_name"),
        vec![("lib.rs".into(), 2)]
    );
    let manifest: Value = serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    assert_eq!(
        manifest["parse_revision"],
        vex::index::parse_cache::CACHE_FORMAT_VERSION
    );

    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&index)
        .unwrap();
    file.set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(1000))
        .unwrap();
    let before = file.metadata().unwrap().modified().unwrap();
    drop(file);
    json(dir.path(), &["update"]);
    assert_eq!(
        std::fs::metadata(index).unwrap().modified().unwrap(),
        before
    );
}

fn seed_semantic_legacy(files: &[(&str, &str)]) -> (TempDir, PathBuf, Vec<Vec<f32>>, Vec<u64>) {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join(".vex.toml"), "semantic = false\n").unwrap();
    for (path, source) in files {
        std::fs::write(dir.path().join(path), source).unwrap();
    }
    let index = index_project(dir.path());
    let mut parsed: Vec<ParsedFile> = files
        .iter()
        .map(|(path, source)| parse_file(path, source, Language::Rust).unwrap())
        .collect();
    let dim = vex::embed::embedder_dim(EMBEDDER).unwrap();
    let budget = vex::embed::embedder_char_budget(EMBEDDER).unwrap();
    let mut vectors = Vec::new();
    let mut hashes = Vec::new();
    let mut tokens = Vec::new();
    for file in &mut parsed {
        for symbol in &file.symbols {
            let vector = one_hot(dim, vectors.len());
            vectors.push(vector);
            let context = vex::embed::build_context(
                symbol.kind.as_str(),
                &symbol.name,
                &file.path,
                symbol.signature.as_deref(),
                symbol.doc.as_deref(),
                symbol.body_tokens.as_deref(),
                budget,
            );
            hashes.push(vex::embed::cache::context_hash(EMBEDDER, &context));
            tokens.push(symbol.body_tokens.clone());
        }
        file.refs.clear();
        file.bound_refs.clear();
    }
    vex::store::writer::write_index_full(&parsed, &vectors, dim, &index).unwrap();
    vex::store::body_tokens::save(&index.with_file_name("index.bodytokens"), &tokens).unwrap();
    vex::index::pipeline::build_hnsw_at(
        &index.with_file_name("index.hnsw"),
        &index.with_file_name("index.hashes"),
        &vectors,
        &hashes,
    )
    .unwrap();

    let manifest_path = index.with_file_name("manifest.json");
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    manifest["parse_revision"] = 5.into();
    manifest["embedder_id"] = EMBEDDER.into();
    manifest["vectors_normalized"] = true.into();
    std::fs::write(manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    (dir, index, vectors, hashes)
}

#[test]
fn semantic_false_migration_retains_verified_vectors_ann_and_embedder() {
    let (dir, index, vectors, hashes) = seed_semantic_legacy(&[(
        "lib.rs",
        "pub fn target_name() {}\nfn caller() { target_name(); }\n",
    )]);
    json(dir.path(), &["update"]);

    let reader = IndexReader::open(&index).unwrap();
    assert_eq!(reader.symbol_count(), vectors.len());
    for (idx, expected) in vectors.iter().enumerate() {
        let record = reader.symbol(idx).unwrap();
        assert_eq!(reader.vector(record.vector_index).unwrap(), expected);
    }
    assert_eq!(
        vex::search::hash_index::load(&index.with_file_name("index.hashes")).unwrap(),
        hashes
    );
    let ann = usearch::new_index(&usearch::IndexOptions {
        dimensions: vectors[0].len(),
        metric: usearch::MetricKind::Cos,
        quantization: usearch::ScalarKind::F32,
        ..Default::default()
    })
    .unwrap();
    ann.view(index.with_file_name("index.hnsw").to_str().unwrap())
        .unwrap();
    assert_eq!(ann.size(), hashes.len());
    assert_eq!(ann.search(&vectors[0], 1).unwrap().keys, [hashes[0]]);
    assert_eq!(
        ordinary_refs(&index, "target_name"),
        vec![("lib.rs".into(), 2)]
    );
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(index.with_file_name("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["embedder_id"], EMBEDDER);
}

#[test]
fn semantic_false_migration_keeps_matching_vectors_before_unembedded_files() {
    let (dir, index, vectors, _) = seed_semantic_legacy(&[
        ("a.rs", "pub fn replaced() {}\n"),
        (
            "b.rs",
            "pub fn keep_value() {}\nfn run() { keep_value(); }\n",
        ),
    ]);
    let expected = HashMap::from([
        ("keep_value".to_string(), vectors[1].clone()),
        ("run".to_string(), vectors[2].clone()),
    ]);
    std::fs::write(dir.path().join("a.rs"), "pub fn changed() {}\n").unwrap();
    std::fs::write(dir.path().join("new.rs"), "pub fn added() {}\n").unwrap();
    json(dir.path(), &["update"]);

    let reader = IndexReader::open(&index).unwrap();
    for idx in 0..reader.symbol_count() {
        let record = reader.symbol(idx).unwrap();
        let path = reader.read_string(record.file_offset);
        let name = reader.read_string(record.name_offset);
        if path == "b.rs" {
            assert_eq!(reader.vector(record.vector_index).unwrap(), &expected[name]);
        } else {
            assert_eq!(record.vector_index, u32::MAX);
        }
    }
    assert!(!index.with_file_name("index.hnsw").exists());
    assert!(!index.with_file_name("index.hashes").exists());
    assert_eq!(
        ordinary_refs(&index, "keep_value"),
        vec![("b.rs".into(), 2)]
    );
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(index.with_file_name("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["embedder_id"], EMBEDDER);
}

#[test]
fn semantic_false_migration_rejects_unverified_vectors() {
    let (dir, index, _, mut hashes) = seed_semantic_legacy(&[(
        "lib.rs",
        "pub fn target_name() {}\nfn caller() { target_name(); }\n",
    )]);
    hashes[0] ^= 1;
    vex::search::hash_index::save(&index.with_file_name("index.hashes"), &hashes).unwrap();
    json(dir.path(), &["update"]);

    let reader = IndexReader::open(&index).unwrap();
    assert!(!reader.has_vectors());
    assert!(!index.with_file_name("index.hnsw").exists());
    assert!(!index.with_file_name("index.hashes").exists());
    assert_eq!(
        ordinary_refs(&index, "target_name"),
        vec![("lib.rs".into(), 2)]
    );
}

fn partial_migration_fixture() -> (TempDir, PathBuf) {
    let (dir, index, _, _) = seed_semantic_legacy(&[
        ("a.rs", "pub fn replaced() {}\n"),
        (
            "b.rs",
            "pub fn keep_value() {}\nfn run() { keep_value(); }\n",
        ),
    ]);
    std::fs::write(dir.path().join("a.rs"), "pub fn changed() {}\n").unwrap();
    std::fs::write(dir.path().join("new.rs"), "pub fn added() {}\n").unwrap();
    json(dir.path(), &["update"]);

    let reader = IndexReader::open(&index).unwrap();
    let covered = (0..reader.symbol_count())
        .filter(|&idx| {
            let record = reader.symbol(idx).unwrap();
            reader.vector(record.vector_index).is_some()
        })
        .count();
    assert_eq!((reader.symbol_count(), covered), (4, 2));
    drop(reader);
    (dir, index)
}

#[test]
fn semantic_update_repairs_partial_vectors_without_source_changes() {
    let (dir, index) = partial_migration_fixture();
    let files = [
        ("a.rs", "pub fn changed() {}\n"),
        (
            "b.rs",
            "pub fn keep_value() {}\nfn run() { keep_value(); }\n",
        ),
        ("new.rs", "pub fn added() {}\n"),
    ];
    let expected = seed_embed_cache(
        &index,
        EMBEDDER,
        &files,
        &[("keep_value", 1), ("run", 2), ("changed", 3), ("added", 4)],
    );

    json(dir.path(), &["update", "--semantic"]);

    assert_vectors_by_name(&index, &expected);
}

#[test]
fn semantic_index_repairs_partial_vectors_without_source_changes() {
    let (dir, index) = partial_migration_fixture();
    let files = [
        ("a.rs", "pub fn changed() {}\n"),
        (
            "b.rs",
            "pub fn keep_value() {}\nfn run() { keep_value(); }\n",
        ),
        ("new.rs", "pub fn added() {}\n"),
    ];
    let expected = seed_embed_cache(
        &index,
        EMBEDDER,
        &files,
        &[("keep_value", 1), ("run", 2), ("changed", 3), ("added", 4)],
    );

    json(
        dir.path(),
        &["index", "--semantic", "--no-clusters", "--no-pattern-index"],
    );

    assert_vectors_by_name(&index, &expected);
}

#[test]
fn semantic_update_rebuilds_partial_vectors_before_merging_changed_files() {
    let (dir, index) = partial_migration_fixture();
    let changed = "pub fn changed_after_semantic_request() {}\n";
    std::fs::write(dir.path().join("a.rs"), changed).unwrap();
    let files = [
        ("a.rs", changed),
        (
            "b.rs",
            "pub fn keep_value() {}\nfn run() { keep_value(); }\n",
        ),
        ("new.rs", "pub fn added() {}\n"),
    ];
    let expected = seed_embed_cache(
        &index,
        EMBEDDER,
        &files,
        &[
            ("keep_value", 1),
            ("run", 2),
            ("changed_after_semantic_request", 3),
            ("added", 4),
        ],
    );

    json(dir.path(), &["update", "--semantic"]);

    assert_vectors_by_name(&index, &expected);
}

#[test]
fn semantic_update_reembeds_all_symbols_when_embedder_changes() {
    let files = [("lib.rs", "pub fn alpha() {}\npub fn beta() {}\n")];
    let (dir, index, _, _) = seed_semantic_legacy(&files);
    json(dir.path(), &["update"]);
    let expected = seed_embed_cache(&index, OTHER_EMBEDDER, &files, &[("alpha", 5), ("beta", 6)]);

    json(
        dir.path(),
        &["update", "--semantic", "--embedder", OTHER_EMBEDDER],
    );

    assert_vectors_by_name(&index, &expected);
    let reader = IndexReader::open(&index).unwrap();
    assert_eq!(
        reader.header().vector_dim,
        vex::embed::embedder_dim(OTHER_EMBEDDER).unwrap()
    );
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(index.with_file_name("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["embedder_id"], OTHER_EMBEDDER);
}

#[test]
fn later_nonsemantic_update_keeps_partial_vector_identity_and_embedder() {
    let (dir, index) = partial_migration_fixture();
    std::fs::write(
        dir.path().join("a.rs"),
        "pub fn changed_again_without_semantic() {}\n",
    )
    .unwrap();
    json(dir.path(), &["update"]);

    let reader = IndexReader::open(&index).unwrap();
    let expected = HashMap::from([
        ("keep_value", one_hot(reader.header().vector_dim, 1)),
        ("run", one_hot(reader.header().vector_dim, 2)),
    ]);
    for idx in 0..reader.symbol_count() {
        let record = reader.symbol(idx).unwrap();
        let name = reader.read_string(record.name_offset);
        match expected.get(name) {
            Some(vector) => assert_eq!(reader.vector(record.vector_index), Some(vector.as_slice())),
            None => assert_eq!(
                record.vector_index,
                u32::MAX,
                "unexpected vector for {name}"
            ),
        }
    }
    drop(reader);
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(index.with_file_name("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["embedder_id"], EMBEDDER);
}

#[test]
fn nonsemantic_changed_update_keeps_complete_index_vector_identity_and_embedder() {
    let (dir, index, vectors, _) = seed_semantic_legacy(&[
        ("a.rs", "pub fn alpha() {}\n"),
        ("b.rs", "pub fn beta() {}\n"),
    ]);
    json(dir.path(), &["update"]);
    std::fs::write(dir.path().join("a.rs"), "pub fn alpha_changed() {}\n").unwrap();

    json(dir.path(), &["update"]);

    let reader = IndexReader::open(&index).unwrap();
    for idx in 0..reader.symbol_count() {
        let record = reader.symbol(idx).unwrap();
        let name = reader.read_string(record.name_offset);
        if name == "beta" {
            assert_eq!(
                reader.vector(record.vector_index),
                Some(vectors[1].as_slice())
            );
        } else {
            assert_eq!(name, "alpha_changed");
            assert_eq!(record.vector_index, u32::MAX);
        }
    }
    drop(reader);
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(index.with_file_name("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["embedder_id"], EMBEDDER);
}
