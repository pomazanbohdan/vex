use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::Result;

use super::*;

static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "vex-query-cache-test-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct FixedEmbedder {
    vector: Vec<f32>,
}

impl Embedder for FixedEmbedder {
    fn id(&self) -> &'static str {
        MINILM_ID
    }
    fn dim(&self) -> u32 {
        384
    }
    fn char_budget(&self) -> usize {
        256
    }
    fn embed(&mut self, _text: &str) -> Result<Vec<f32>> {
        Ok(self.vector.clone())
    }
    fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|text| self.embed(text)).collect()
    }
}

fn unit_vector() -> Vec<f32> {
    let mut vector = vec![0.0; 384];
    vector[0] = 1.0;
    vector
}

fn test_embedder(vector: Vec<f32>) -> Box<dyn Embedder> {
    Box::new(FixedEmbedder { vector })
}

#[test]
fn query_key_tracks_only_semantic_inputs() {
    let dir = TestDir::new();
    let key = QueryKey::new(MINILM_ID, false, "same query");
    assert_eq!(
        key.path(dir.path()),
        QueryKey::new(MINILM_ID, false, "same query").path(dir.path())
    );
    assert_ne!(
        key.path(dir.path()),
        QueryKey::new(MINILM_ID, true, "same query").path(dir.path())
    );
    assert_ne!(
        key.path(dir.path()),
        QueryKey::new(MINILM_ID, false, "different query").path(dir.path())
    );
    assert_ne!(
        key.path(dir.path()),
        QueryKey::new("other-model", false, "same query").path(dir.path())
    );
    let base = QueryKey::from_parts(
        b"query-vectors-v2",
        b"model",
        b"sha",
        b"version",
        false,
        b"query",
    );
    for variant in [
        QueryKey::from_parts(
            b"other-format",
            b"model",
            b"sha",
            b"version",
            false,
            b"query",
        ),
        QueryKey::from_parts(
            b"query-vectors-v2",
            b"model",
            b"other-sha",
            b"version",
            false,
            b"query",
        ),
        QueryKey::from_parts(
            b"query-vectors-v2",
            b"model",
            b"sha",
            b"other-version",
            false,
            b"query",
        ),
    ] {
        assert_ne!(base.path(dir.path()), variant.path(dir.path()));
    }
    assert_ne!(
        QueryKey::from_parts(b"ab", b"c", b"sha", b"version", false, b"query").path(dir.path()),
        QueryKey::from_parts(b"a", b"bc", b"sha", b"version", false, b"query").path(dir.path())
    );
}

#[test]
fn cache_hit_preserves_exact_float_bits_and_skips_factory() {
    let dir = TestDir::new();
    let cache = RawQueryCache::new(dir.path());
    let key = QueryKey::new(MINILM_ID, false, "query");
    let mut vector = unit_vector();
    vector[1] = f32::from_bits(0x8000_0000);
    assert!(cache.put(&key, &vector));
    let calls = Arc::new(AtomicUsize::new(0));
    let mut embedder = make_query_embedder_with(
        MINILM_ID,
        false,
        true,
        cache,
        {
            let calls = Arc::clone(&calls);
            move || {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(test_embedder(unit_vector()))
            }
        },
        || Ok(true),
    )
    .unwrap();
    let got = embedder.embed("query").unwrap();
    assert_eq!(
        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        vector.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn gpu_and_skip_check_bypass_cache() {
    for (cpu, allowed) in [(false, true), (true, false)] {
        let dir = TestDir::new();
        let key = QueryKey::new(MINILM_ID, false, "query");
        let mut cached = vec![0.0; 384];
        cached[1] = 1.0;
        fs::write(key.path(dir.path()), encode(&cached)).unwrap();
        let blocked_path = dir.path().join("not-a-directory");
        fs::write(&blocked_path, b"sentinel").unwrap();
        let cache = RawQueryCache::new(dir.path());
        let eligibility_calls = Arc::new(AtomicUsize::new(0));
        let mut embedder = make_query_embedder_with(
            MINILM_ID,
            false,
            cpu,
            cache,
            || Ok(test_embedder(unit_vector())),
            {
                let eligibility_calls = Arc::clone(&eligibility_calls);
                move || {
                    eligibility_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(allowed)
                }
            },
        )
        .unwrap();
        assert_eq!(embedder.embed("query").unwrap(), unit_vector());
        assert_eq!(fs::read(&blocked_path).unwrap(), b"sentinel");
        assert_eq!(fs::read(key.path(dir.path())).unwrap(), encode(&cached));
        let expected_checks = if cpu { 2 } else { 0 };
        assert_eq!(eligibility_calls.load(Ordering::SeqCst), expected_checks);
    }
}

#[test]
fn raw_cache_rejects_bad_length_nan_infinity_and_norm() {
    let dir = TestDir::new();
    let cache = RawQueryCache::new(dir.path());
    let valid_key = QueryKey::new(MINILM_ID, false, "valid");
    let valid = unit_vector();
    fs::write(valid_key.path(dir.path()), encode(&valid)).unwrap();
    assert_eq!(cache.get(&valid_key), Some(valid.try_into().unwrap()));
    for (i, bytes) in [
        vec![0; VECTOR_BYTES - 1],
        vec![0; VECTOR_BYTES + 1],
        {
            let mut v = unit_vector();
            v[0] = f32::NAN;
            encode(&v)
        },
        {
            let mut v = unit_vector();
            v[0] = f32::INFINITY;
            encode(&v)
        },
        {
            let mut v = vec![0.0; 384];
            v[0] = 0.5;
            encode(&v)
        },
    ]
    .into_iter()
    .enumerate()
    {
        let key = QueryKey::new(MINILM_ID, false, &i.to_string());
        fs::write(key.path(dir.path()), bytes).unwrap();
        assert!(cache.get(&key).is_none());
    }
}

#[test]
fn failed_integrity_never_populates_cache() {
    let dir = TestDir::new();
    let cache = RawQueryCache::new(dir.path());
    let key = QueryKey::new(MINILM_ID, false, "query");
    let mut embedder = make_query_embedder_with(
        MINILM_ID,
        false,
        true,
        cache,
        || Ok(test_embedder(unit_vector())),
        || Ok(false),
    )
    .unwrap();
    assert_eq!(embedder.embed("query").unwrap(), unit_vector());
    assert!(!key.path(dir.path()).exists());
}

#[test]
fn failed_integrity_error_propagates_without_cache_write() {
    let dir = TestDir::new();
    let key = QueryKey::new(MINILM_ID, false, "query");
    let mut embedder = make_query_embedder_with(
        MINILM_ID,
        false,
        true,
        RawQueryCache::new(dir.path()),
        || Ok(test_embedder(unit_vector())),
        || Err(anyhow::anyhow!("integrity failed")),
    )
    .unwrap();
    assert!(embedder
        .embed("query")
        .unwrap_err()
        .to_string()
        .contains("integrity failed"));
    assert!(!key.path(dir.path()).exists());
}

#[test]
fn post_factory_integrity_recheck_prevents_store() {
    let dir = TestDir::new();
    let key = QueryKey::new(MINILM_ID, false, "query");
    let checks = Arc::new(AtomicUsize::new(0));
    let mut embedder = make_query_embedder_with(
        MINILM_ID,
        false,
        true,
        RawQueryCache::new(dir.path()),
        || Ok(test_embedder(unit_vector())),
        {
            let checks = Arc::clone(&checks);
            move || Ok(checks.fetch_add(1, Ordering::SeqCst) == 0)
        },
    )
    .unwrap();
    assert_eq!(embedder.embed("query").unwrap(), unit_vector());
    assert!(!key.path(dir.path()).exists());
}

#[test]
fn concurrent_same_key_writes_remain_valid() {
    let dir = TestDir::new();
    let cache = RawQueryCache::new(dir.path());
    let key = QueryKey::new(MINILM_ID, false, "query");
    let expected = encode(&unit_vector());
    fs::write(key.path(dir.path()), &expected).unwrap();
    let (publishing, active) = std::sync::mpsc::channel();
    let (observed, read_done) = std::sync::mpsc::channel();
    let writer_cache = RawQueryCache::new(dir.path());
    let writer_key = key.clone();
    let mut writer_expected = vec![0.0; 384];
    writer_expected[1] = 1.0;
    let published = writer_expected.clone();
    let writer = std::thread::spawn(move || {
        assert!(
            writer_cache.put_with(&writer_key, &writer_expected, |temp, destination| {
                publishing.send(()).unwrap();
                read_done.recv().unwrap();
                fs::rename(temp, destination)
            })
        );
    });
    active.recv().unwrap();
    let mut bytes = Vec::with_capacity(VECTOR_BYTES + 1);
    File::open(key.path(dir.path()))
        .unwrap()
        .take((VECTOR_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .unwrap();
    assert_eq!(bytes.len(), VECTOR_BYTES);
    assert_eq!(bytes, expected);
    assert_eq!(decode(&bytes), Some(unit_vector().try_into().unwrap()));
    observed.send(()).unwrap();
    writer.join().unwrap();
    assert_eq!(
        cache.get(&key),
        Some(<[f32; 384]>::try_from(published).unwrap())
    );
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let cache = RawQueryCache::new(dir.path());
            let key = key.clone();
            std::thread::spawn(move || {
                for _ in 0..20 {
                    assert!(cache.put(&key, &unit_vector()));
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
    assert_eq!(cache.get(&key), Some(unit_vector().try_into().unwrap()));
}

#[test]
fn concurrent_distinct_key_stores_converge_to_file_bound() {
    let dir = TestDir::new();
    let workers: Vec<_> = (0..8)
        .map(|worker| {
            let directory = dir.path().to_path_buf();
            std::thread::spawn(move || {
                let cache = RawQueryCache::new(directory);
                for index in 0..40 {
                    let key = QueryKey::new(MINILM_ID, false, &format!("{worker}-{index}"));
                    assert!(cache.put(&key, &unit_vector()));
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    let count = fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "bin")
        })
        .count();
    assert!(count > 0 && count <= MAX_FILES);
}

#[test]
fn successful_store_prunes_to_exact_file_limit() {
    let dir = TestDir::new();
    for index in 0..=MAX_FILES {
        fs::write(dir.path().join(format!("old-{index:03}.bin")), b"old").unwrap();
    }
    let cache = RawQueryCache::new(dir.path());
    let key = QueryKey::new(MINILM_ID, false, "new entry");
    assert!(cache.put(&key, &unit_vector()));
    assert_eq!(
        fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "bin"))
            .count(),
        MAX_FILES
    );
}

#[test]
fn read_only_cache_directory_preserves_search() {
    let dir = TestDir::new();
    let blocked = dir.path().join("file");
    fs::write(&blocked, b"still here").unwrap();
    let mut embedder = make_query_embedder_with(
        MINILM_ID,
        false,
        true,
        RawQueryCache::new(&blocked),
        || Ok(test_embedder(unit_vector())),
        || Ok(true),
    )
    .unwrap();
    assert_eq!(embedder.embed("query").unwrap(), unit_vector());
    assert_eq!(fs::read(blocked).unwrap(), b"still here");
}

#[test]
fn rename_error_with_valid_winner_preserves_vector() {
    let dir = TestDir::new();
    let cache = RawQueryCache::new(dir.path());
    let key = QueryKey::new(MINILM_ID, false, "query");
    let expected = unit_vector();
    let foreign_slot = dir.path().join(".tmp-slot-31");
    fs::write(&foreign_slot, b"another writer's slot").unwrap();
    let mut owned_temp = None;
    assert!(cache.put_with(&key, &expected, |temp, destination| {
        owned_temp = Some(temp.to_path_buf());
        fs::write(destination, encode(&expected))?;
        Err(std::io::Error::other(
            "simulated rename failure after winner publication",
        ))
    }));
    assert!(!owned_temp.unwrap().exists());
    assert_eq!(fs::read(foreign_slot).unwrap(), b"another writer's slot");
    assert_eq!(cache.get(&key), Some(unit_vector().try_into().unwrap()));
}

#[test]
fn rename_error_with_corrupt_destination_never_unlinks_foreign_file() {
    let dir = TestDir::new();
    let cache = RawQueryCache::new(dir.path());
    let key = QueryKey::new(MINILM_ID, false, "query");
    fs::write(key.path(dir.path()), b"foreign corrupt entry").unwrap();
    let foreign_slot = dir.path().join(".tmp-slot-31");
    fs::write(&foreign_slot, b"owned by another writer").unwrap();
    assert!(
        !cache.put_with(&key, &unit_vector(), |_owned_temp, _destination| {
            Err(std::io::Error::other("simulated rename failure"))
        })
    );
    assert_eq!(
        fs::read(key.path(dir.path())).unwrap(),
        b"foreign corrupt entry"
    );
    assert!(!dir.path().join(".tmp-slot-00").exists());
    assert_eq!(fs::read(foreign_slot).unwrap(), b"owned by another writer");
    assert!(cache.put(&key, &unit_vector()));
    assert_eq!(cache.get(&key), Some(unit_vector().try_into().unwrap()));
}

#[test]
fn abandoned_temp_slots_have_a_fixed_bound() {
    let dir = TestDir::new();
    let cache = RawQueryCache::new(dir.path());
    let slot_bytes = b"abandoned slot remains byte-identical";
    for slot in 0..MAX_TEMP_FILES {
        fs::write(dir.path().join(format!(".tmp-slot-{slot:02}")), slot_bytes).unwrap();
    }
    let mut embedder = make_query_embedder_with(
        MINILM_ID,
        false,
        true,
        cache,
        || Ok(test_embedder(unit_vector())),
        || Ok(true),
    )
    .unwrap();
    assert_eq!(embedder.embed("uncached query").unwrap(), unit_vector());
    assert!(!dir.path().join(".tmp-slot-32").exists());
    for slot in 0..MAX_TEMP_FILES {
        assert_eq!(
            fs::read(dir.path().join(format!(".tmp-slot-{slot:02}"))).unwrap(),
            slot_bytes
        );
    }
    assert!(!QueryKey::new(MINILM_ID, false, "uncached query")
        .path(dir.path())
        .exists());
    assert_eq!(
        fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "bin"))
            .count(),
        0
    );
}

fn encode(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
