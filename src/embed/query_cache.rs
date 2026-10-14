use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result};
use fs2::FileExt;
use sha2::{Digest, Sha256};

use super::{cache::EmbedCache, Embedder, MINILM_CHAR_BUDGET, MINILM_DIM, MINILM_ID};

const REPO: &str = "models--Qdrant--all-MiniLM-L6-v2-onnx";
const FILES: [&str; 5] = [
    "model.onnx",
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
];
const MAX_FILES: usize = 256;
const MAX_BYTES: u64 = 8 * 1024 * 1024;

pub(crate) fn make_query_embedder(id: &str, normalized: bool) -> Result<Box<dyn Embedder>> {
    if id != MINILM_ID {
        return super::make_embedder(id).context("load embedding model");
    }
    Ok(Box::new(CachedQueryEmbedder {
        directory: crate::util::config::embed_cache_dir().join("query-vectors"),
        hash_current_artifacts: Box::new(move || hash_current_artifacts(normalized)),
        factory: Box::new(|| super::make_embedder(MINILM_ID).context("load embedding model")),
    }))
}

fn hash_current_artifacts(normalized: bool) -> Option<String> {
    if std::env::var("VEX_EMBEDDER_SKIP_CHECK").ok().as_deref() == Some("1") {
        return None;
    }
    // fastembed's pull_from_hf uses HF_HOME directly, without /hub.
    let root = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| crate::util::config::embed_cache_dir());
    artifact_identity(&root, super::integrity::MINILM_ONNX_SHA256, normalized)
        .ok()
        .flatten()
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn artifact_identity(root: &Path, pin: &str, normalized: bool) -> Result<Option<String>> {
    let repo = root.join(REPO);
    let reference = repo.join("refs/main");
    let commit = fs::read(&reference)?;
    if commit.len() != 40 || !commit.iter().all(u8::is_ascii_hexdigit) {
        return Ok(None);
    }
    let snapshot = repo.join("snapshots").join(std::str::from_utf8(&commit)?);
    let mut identity = Sha256::new();
    identity.update(format!(
        "query-v1;vex={};model={MINILM_ID};dim={MINILM_DIM};CPU;arch={};maxlen=512;pool=Mean;quant=None;output=default;l2=fastembed;search-normalized={normalized};threads={};endpoint={}",
        env!("CARGO_PKG_VERSION"), std::env::consts::ARCH,
        std::thread::available_parallelism()?.get(),
        std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into())
    ));
    let dependency_lock = include_bytes!("../../Cargo.lock");
    identity.update((dependency_lock.len() as u64).to_le_bytes());
    identity.update(dependency_lock);
    identity.update(&commit);
    for name in FILES {
        let path = snapshot.join(name).canonicalize()?;
        let mut file = File::open(&path)?;
        let before = file.metadata()?;
        if !before.is_file() {
            return Ok(None);
        }
        let modified = before.modified()?;
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hash.update(&buffer[..n]);
        }
        let sha = format!("{:x}", hash.finalize());
        if name == "model.onnx" && sha != pin {
            return Ok(None);
        }
        let after = file.metadata()?;
        if before.len() != after.len() || modified != after.modified()? {
            return Ok(None);
        }
        let Some(path_text) = path.to_str() else {
            return Ok(None);
        };
        for field in [
            name.to_owned(),
            path_text.to_owned(),
            sha,
            format!("{}:{modified:?}", before.len()),
        ] {
            identity.update((field.len() as u64).to_le_bytes());
            identity.update(field.as_bytes());
        }
    }
    if fs::read(reference)? != commit {
        return Ok(None);
    }
    Ok(Some(format!("{:x}", identity.finalize())))
}

type Factory = Box<dyn FnMut() -> Result<Box<dyn Embedder>> + Send>;
type ArtifactHash = Box<dyn Fn() -> Option<String> + Send>;

struct CachedQueryEmbedder {
    directory: PathBuf,
    hash_current_artifacts: ArtifactHash,
    factory: Factory,
}

fn query_id(identity: &str, text: &str) -> String {
    format!("query-v1:{identity}:{}", digest(text.as_bytes()))
}

fn cache_path(directory: &Path, id: &str) -> PathBuf {
    directory.join(format!("{}.bin", digest(id.as_bytes())))
}

fn valid_vector(vector: &[f32]) -> bool {
    if vector.len() != MINILM_DIM as usize || vector.iter().any(|v| !v.is_finite()) {
        return false;
    }
    let norm: f64 = vector.iter().map(|&v| f64::from(v).powi(2)).sum();
    (norm - 1.0).abs() <= 0.001
}

fn load(directory: &Path, id: &str) -> Option<Vec<f32>> {
    let path = cache_path(directory, id);
    let expected = 28 + id.len() + MINILM_DIM as usize * 4;
    if fs::metadata(&path).ok()?.len() != expected as u64 {
        return None;
    }
    let cache = EmbedCache::load_with_entry_limit(&path, id, MINILM_DIM, 1);
    if cache.len() != 1 {
        return None;
    }
    let vector = cache.get(0)?;
    valid_vector(vector).then(|| vector.to_vec())
}

fn store(directory: &Path, id: &str, vector: &[f32]) -> Result<()> {
    fs::create_dir_all(directory)?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(directory.join("query-vectors.lock"))?;
    lock.try_lock_exclusive()?;
    prune(directory)?;
    let mut cache = EmbedCache::empty(id, MINILM_DIM);
    cache.insert(0, vector.to_vec());
    let path = cache_path(directory, id);
    if let Err(error) = cache.save(&path) {
        let _ = fs::remove_file(path.with_extension("bin.tmp"));
        return Err(error);
    }
    if let Err(error) = prune(directory) {
        let _ = fs::remove_file(path);
        return Err(error);
    }
    Ok(()) // Closing the file releases the fs2 lock, including on errors.
}

fn prune(directory: &Path) -> Result<()> {
    let mut files = Vec::new();
    let mut bytes = 0u64;
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if entry.file_type()?.is_file() && entry.file_name().to_string_lossy().ends_with(".bin.tmp")
        {
            fs::remove_file(entry.path())?;
            continue;
        }
        if entry.path().extension().and_then(|s| s.to_str()) != Some("bin")
            || !entry.file_type()?.is_file()
        {
            continue;
        }
        let metadata = entry.metadata()?;
        bytes = bytes.saturating_add(metadata.len());
        files.push((
            metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            entry.path(),
            metadata.len(),
        ));
    }
    files.sort_unstable();
    let mut count = files.len();
    for (_, path, size) in files {
        if count <= MAX_FILES && bytes <= MAX_BYTES {
            break;
        }
        fs::remove_file(path)?;
        count -= 1;
        bytes = bytes.saturating_sub(size);
    }
    Ok(())
}

impl Embedder for CachedQueryEmbedder {
    fn id(&self) -> &'static str {
        MINILM_ID
    }
    fn dim(&self) -> u32 {
        MINILM_DIM
    }
    fn char_budget(&self) -> usize {
        MINILM_CHAR_BUDGET
    }
    fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let before = (self.hash_current_artifacts)();
        if let Some(identity) = &before {
            let id = query_id(identity, text);
            if let Some(vector) = load(&self.directory, &id) {
                if (self.hash_current_artifacts)().as_ref() == Some(identity) {
                    tracing::debug!("query vector cache: hit before factory");
                    return Ok(vector);
                }
            }
        }
        tracing::debug!("query vector cache: create fresh factory");
        let vector = (self.factory)()?.embed(text)?;
        if let Some(identity) = before {
            if valid_vector(&vector) && (self.hash_current_artifacts)().as_ref() == Some(&identity)
            {
                if let Err(error) = store(&self.directory, &query_id(&identity, text), &vector) {
                    tracing::debug!(%error, "query vector cache: save failed");
                }
            }
        }
        Ok(vector)
    }
    fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|text| self.embed(text)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct Fake;
    impl Embedder for Fake {
        fn id(&self) -> &'static str {
            MINILM_ID
        }
        fn dim(&self) -> u32 {
            MINILM_DIM
        }
        fn char_budget(&self) -> usize {
            MINILM_CHAR_BUDGET
        }
        fn embed(&mut self, _: &str) -> Result<Vec<f32>> {
            let mut vector = vec![0.0; MINILM_DIM as usize];
            vector[0] = f32::from_bits(0x3f800001);
            Ok(vector)
        }
        fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            texts.iter().map(|text| self.embed(text)).collect()
        }
    }

    fn cached(directory: PathBuf, calls: Arc<AtomicUsize>) -> CachedQueryEmbedder {
        CachedQueryEmbedder {
            directory,
            hash_current_artifacts: Box::new(|| Some("a".repeat(64))),
            factory: Box::new(move || {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(Box::new(Fake))
            }),
        }
    }

    #[test]
    fn repeated_query_bypasses_factory_preserves_raw_bits() {
        let temp = tempfile::tempdir().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let mut first = cached(temp.path().to_owned(), calls.clone());
        let expected = first.embed(" Exact Query ").unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let mut second = cached(temp.path().to_owned(), calls.clone());
        second.factory = Box::new(|| panic!("query cache hit constructed model"));
        let actual = second.embed(" Exact Query ").unwrap();
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn exact_utf8_query_and_identity_select_vectors() {
        let temp = tempfile::tempdir().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let mut embedder = cached(temp.path().to_owned(), calls.clone());
        for query in ["Query", "query", " Query", "Query ", "Код"] {
            embedder.embed(query).unwrap();
            embedder.embed(query).unwrap();
        }
        assert_eq!(calls.load(Ordering::SeqCst), 5);
        embedder.hash_current_artifacts = Box::new(|| Some("b".repeat(64)));
        embedder.embed("Query").unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 6);
    }

    fn artifact_fixture(root: &Path) -> (PathBuf, String) {
        let repo = root.join(REPO);
        fs::create_dir_all(repo.join("refs")).unwrap();
        let commit = "a".repeat(40);
        fs::write(repo.join("refs/main"), &commit).unwrap();
        let snapshot = repo.join("snapshots").join(commit);
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(snapshot.join("model.onnx"), b"model123").unwrap();
        for name in &FILES[1..] {
            fs::write(snapshot.join(name), b"{}").unwrap();
        }
        (snapshot, digest(b"model123"))
    }

    #[test]
    fn artifact_bytes_ref_layout_and_options_invalidate_identity() {
        let temp = tempfile::tempdir().unwrap();
        let (snapshot, pin) = artifact_fixture(temp.path());
        let identity = artifact_identity(temp.path(), &pin, false)
            .unwrap()
            .unwrap();
        assert_ne!(
            Some(identity.clone()),
            artifact_identity(temp.path(), &pin, true).unwrap()
        );
        for name in &FILES[1..] {
            let path = snapshot.join(name);
            let modified =
                filetime::FileTime::from_last_modification_time(&fs::metadata(&path).unwrap());
            fs::write(&path, b"[]").unwrap();
            filetime::set_file_mtime(&path, modified).unwrap();
            assert_ne!(
                Some(identity.clone()),
                artifact_identity(temp.path(), &pin, false).unwrap()
            );
            fs::write(&path, b"{}").unwrap();
            filetime::set_file_mtime(&path, modified).unwrap();
        }
        let changed = temp
            .path()
            .join(REPO)
            .join("snapshots")
            .join("b".repeat(40));
        fs::create_dir(&changed).unwrap();
        for name in FILES {
            fs::copy(snapshot.join(name), changed.join(name)).unwrap();
        }
        fs::write(temp.path().join(REPO).join("refs/main"), "b".repeat(40)).unwrap();
        assert_ne!(
            Some(identity),
            artifact_identity(temp.path(), &pin, false).unwrap()
        );
    }

    #[test]
    fn same_metadata_model_replacement_never_hits() {
        let temp = tempfile::tempdir().unwrap();
        let (snapshot, pin) = artifact_fixture(temp.path());
        assert!(artifact_identity(temp.path(), &pin, false)
            .unwrap()
            .is_some());
        let path = snapshot.join("model.onnx");
        let before = fs::metadata(&path).unwrap();
        fs::write(&path, b"tampered").unwrap();
        filetime::set_file_mtime(
            &path,
            filetime::FileTime::from_last_modification_time(&before),
        )
        .unwrap();
        let after = fs::metadata(&path).unwrap();
        assert_eq!(before.len(), after.len());
        assert_eq!(before.modified().unwrap(), after.modified().unwrap());
        assert!(artifact_identity(temp.path(), &pin, false)
            .unwrap()
            .is_none());
    }

    #[test]
    fn missing_bundle_and_invalid_ref_use_fresh_factory() {
        let temp = tempfile::tempdir().unwrap();
        let (_, pin) = artifact_fixture(temp.path());
        let root = temp.path().to_owned();
        let calls = Arc::new(AtomicUsize::new(0));
        let mut embedder = cached(root.join("query-vectors"), calls.clone());
        let missing = root
            .join(REPO)
            .join("snapshots")
            .join("a".repeat(40))
            .join("config.json");
        fs::remove_file(missing).unwrap();
        embedder.hash_current_artifacts =
            Box::new(move || artifact_identity(&root, &pin, false).ok().flatten());
        embedder.embed("query").unwrap();
        embedder.embed("query").unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        fs::write(temp.path().join(REPO).join("refs/main"), b"../outside").unwrap();
        assert!(artifact_identity(temp.path(), &digest(b"model123"), false)
            .unwrap()
            .is_none());
    }

    #[test]
    fn identity_change_around_hit_or_inference_prevents_reuse() {
        let temp = tempfile::tempdir().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let mut embedder = cached(temp.path().to_owned(), calls.clone());
        embedder.embed("query").unwrap();
        let probes = Arc::new(AtomicUsize::new(0));
        let probe = probes.clone();
        embedder.hash_current_artifacts = Box::new(move || {
            Some(
                if probe.fetch_add(1, Ordering::SeqCst) == 0 {
                    "a"
                } else {
                    "b"
                }
                .repeat(64),
            )
        });
        embedder.embed("query").unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(load(temp.path(), &query_id(&"b".repeat(64), "query")).is_none());
    }

    #[test]
    fn cached_query_runs_current_retrieval_after_source_reindex() {
        use crate::{
            parse::{language::Language, parse_file},
            search::semantic::search_with_embedder,
            store::{reader::IndexReader, writer::write_index_full},
        };
        let temp = tempfile::tempdir().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let mut embedder = cached(temp.path().join("queries"), calls.clone());
        let source = temp.path().join("fixture.rs");
        let index = temp.path().join("index.vex");
        let hnsw = temp.path().join("index.hnsw");
        let vector = Fake.embed("query").unwrap();
        for name in ["before", "after"] {
            fs::write(&source, format!("fn {name}() {{}}\n")).unwrap();
            let parsed = parse_file(
                "fixture.rs",
                &fs::read_to_string(&source).unwrap(),
                Language::Rust,
            )
            .unwrap();
            assert_eq!(parsed.symbols.len(), 1);
            write_index_full(&[parsed], std::slice::from_ref(&vector), MINILM_DIM, &index).unwrap();
            let reader = IndexReader::open(&index).unwrap();
            let result =
                search_with_embedder(&reader, &mut embedder, "query", 1, &hnsw, true).unwrap();
            assert_eq!(result.len(), 1);
            assert_eq!(result[0].name, name);
            assert_eq!((result[0].path.as_str(), result[0].line), ("fixture.rs", 1));
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn fresh_factory_errors_and_invalid_outputs_keep_original_result() {
        let temp = tempfile::tempdir().unwrap();
        let mut embedder = cached(temp.path().to_owned(), Arc::new(AtomicUsize::new(0)));
        embedder.factory = Box::new(|| anyhow::bail!("original factory failure"));
        assert_eq!(
            embedder.embed("query").unwrap_err().to_string(),
            "original factory failure"
        );
        struct Invalid;
        impl Embedder for Invalid {
            fn id(&self) -> &'static str {
                MINILM_ID
            }
            fn dim(&self) -> u32 {
                MINILM_DIM
            }
            fn char_budget(&self) -> usize {
                MINILM_CHAR_BUDGET
            }
            fn embed(&mut self, _: &str) -> Result<Vec<f32>> {
                Ok(vec![f32::NAN])
            }
            fn embed_batch(&mut self, _: &[String]) -> Result<Vec<Vec<f32>>> {
                unreachable!()
            }
        }
        embedder.factory = Box::new(|| Ok(Box::new(Invalid)));
        assert!(embedder.embed("query").unwrap()[0].is_nan());
        assert!(load(temp.path(), &query_id(&"a".repeat(64), "query")).is_none());
    }

    #[test]
    #[serial_test::serial]
    fn explicit_integrity_bypass_and_unknown_model_keep_fresh_behavior() {
        struct Restore(Option<std::ffi::OsString>);
        impl Drop for Restore {
            fn drop(&mut self) {
                match &self.0 {
                    Some(value) => std::env::set_var("VEX_EMBEDDER_SKIP_CHECK", value),
                    None => std::env::remove_var("VEX_EMBEDDER_SKIP_CHECK"),
                }
            }
        }
        let _restore = Restore(std::env::var_os("VEX_EMBEDDER_SKIP_CHECK"));
        std::env::set_var("VEX_EMBEDDER_SKIP_CHECK", "1");
        assert!(hash_current_artifacts(false).is_none());
        let error = make_query_embedder("unknown-embedder", false)
            .err()
            .unwrap();
        assert!(format!("{error:#}").contains("unknown embedder"));
    }

    #[test]
    fn corrupt_cache_formats_and_vectors_are_misses() {
        let temp = tempfile::tempdir().unwrap();
        let id = query_id(&"a".repeat(64), "query");
        let vector = Fake.embed("query").unwrap();
        store(temp.path(), &id, &vector).unwrap();
        let path = cache_path(temp.path(), &id);
        let bytes = fs::read(&path).unwrap();
        let body = 28 + id.len();
        let mut corruptions = Vec::new();
        for offset in [0, 4, 8, 12, 12 + id.len(), 16 + id.len(), 20 + id.len()] {
            let mut corrupt = bytes.clone();
            corrupt[offset] ^= 0xff;
            corruptions.push(corrupt);
        }
        for value in [f32::NAN, f32::INFINITY, 0.0, 2.0] {
            let mut corrupt = bytes.clone();
            corrupt[body..body + 4].copy_from_slice(&value.to_le_bytes());
            corruptions.push(corrupt);
        }
        let mut extra = bytes.clone();
        extra.extend_from_slice(&[0u8; 1544]);
        corruptions.push(extra);
        corruptions.push(bytes[..bytes.len() - 1].to_vec());
        let mut oversized = bytes.clone();
        oversized.resize(8 * 1024 * 1024 + 1, 0);
        corruptions.push(oversized);
        for (case, corrupt) in corruptions.iter().enumerate() {
            fs::write(&path, corrupt).unwrap();
            assert!(load(temp.path(), &id).is_none(), "corruption case {case}");
        }
        assert!(!valid_vector(&vector[..383]));
    }

    #[test]
    fn unwritable_cache_and_busy_lock_preserve_fresh_results() {
        let temp = tempfile::tempdir().unwrap();
        let blocked = temp.path().join("not-a-directory");
        fs::write(&blocked, b"file").unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let mut embedder = cached(blocked, calls.clone());
        assert_eq!(embedder.embed("query").unwrap()[0].to_bits(), 0x3f800001);
        embedder.directory = temp.path().to_owned();
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(temp.path().join("query-vectors.lock"))
            .unwrap();
        lock.lock_exclusive().unwrap();
        assert_eq!(embedder.embed("query").unwrap()[0].to_bits(), 0x3f800001);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(load(temp.path(), &query_id(&"a".repeat(64), "query")).is_none());
    }

    #[test]
    fn cache_count_and_byte_bounds_and_concurrent_writers() {
        let temp = tempfile::tempdir().unwrap();
        let vector = Fake.embed("query").unwrap();
        for query in 0..257 {
            store(
                temp.path(),
                &query_id(&"a".repeat(64), &query.to_string()),
                &vector,
            )
            .unwrap();
        }
        let files = || {
            fs::read_dir(temp.path())
                .unwrap()
                .map(Result::unwrap)
                .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("bin"))
                .collect::<Vec<_>>()
        };
        assert_eq!(files().len(), MAX_FILES);
        fs::write(
            temp.path().join("oversized.bin"),
            vec![0u8; MAX_BYTES as usize + 1],
        )
        .unwrap();
        fs::write(temp.path().join("orphan.bin.tmp"), b"partial write").unwrap();
        store(temp.path(), &query_id(&"a".repeat(64), "retained"), &vector).unwrap();
        assert!(!temp.path().join("orphan.bin.tmp").exists());
        assert!(files().len() <= MAX_FILES);
        assert!(
            files()
                .iter()
                .map(|e| e.metadata().unwrap().len())
                .sum::<u64>()
                <= MAX_BYTES
        );
        let id = query_id(&"a".repeat(64), "concurrent");
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let id = &id;
                let vector = &vector;
                let directory = temp.path();
                scope.spawn(move || {
                    let _ = store(directory, id, vector);
                });
            }
        });
        assert_eq!(load(temp.path(), &id).unwrap(), vector);
        assert!(files().len() <= MAX_FILES);
    }

    #[cfg(windows)]
    #[test]
    fn readonly_lock_file_keeps_fresh_vector() {
        let temp = tempfile::tempdir().unwrap();
        let lock = temp.path().join("query-vectors.lock");
        fs::write(&lock, b"").unwrap();
        struct Restore(PathBuf, fs::Permissions);
        impl Drop for Restore {
            fn drop(&mut self) {
                let _ = fs::set_permissions(&self.0, self.1.clone());
            }
        }
        let permissions = fs::metadata(&lock).unwrap().permissions();
        let _restore = Restore(lock.clone(), permissions.clone());
        let mut readonly = permissions;
        readonly.set_readonly(true);
        fs::set_permissions(&lock, readonly).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let mut embedder = cached(temp.path().to_owned(), calls.clone());
        assert_eq!(embedder.embed("query").unwrap()[0].to_bits(), 0x3f800001);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(load(temp.path(), &query_id(&"a".repeat(64), "query")).is_none());
    }

    #[cfg(windows)]
    #[test]
    fn in_use_eviction_target_keeps_cache_bounded() {
        use std::os::windows::fs::OpenOptionsExt;
        let temp = tempfile::tempdir().unwrap();
        let vector = Fake.embed("query").unwrap();
        for query in 0..MAX_FILES {
            store(
                temp.path(),
                &query_id(&"a".repeat(64), &query.to_string()),
                &vector,
            )
            .unwrap();
        }
        let protected = cache_path(temp.path(), &query_id(&"a".repeat(64), "0"));
        filetime::set_file_mtime(&protected, filetime::FileTime::from_unix_time(0, 0)).unwrap();
        // FILE_SHARE_READ | FILE_SHARE_WRITE, deliberately excluding DELETE.
        let _held = OpenOptions::new()
            .read(true)
            .share_mode(3)
            .open(&protected)
            .unwrap();
        assert!(
            fs::remove_file(&protected).is_err(),
            "fixture must deny deletion"
        );
        let calls = Arc::new(AtomicUsize::new(0));
        let mut embedder = cached(temp.path().to_owned(), calls.clone());
        assert_eq!(embedder.embed("new-query").unwrap(), vector);
        assert!(load(temp.path(), &query_id(&"a".repeat(64), "new-query")).is_none());
        assert_eq!(
            fs::read_dir(temp.path())
                .unwrap()
                .filter(|e| e
                    .as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .and_then(|s| s.to_str())
                    == Some("bin"))
                .count(),
            MAX_FILES
        );
    }
}
