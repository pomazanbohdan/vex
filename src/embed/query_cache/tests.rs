use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

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

fn unit_vector() -> Vec<f32> {
    let mut vector = vec![0.0; 384];
    vector[0] = 1.0;
    vector
}

fn encode(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn key_path(directory: &Path, id: &str, normalized: bool, query: &str) -> PathBuf {
    let normalized = [u8::from(normalized)];
    directory.join(format!(
        "{}.bin",
        cache_key([
            FORMAT_TAG,
            id.as_bytes(),
            super::super::integrity::MINILM_ONNX_SHA256.as_bytes(),
            env!("CARGO_PKG_VERSION").as_bytes(),
            &normalized,
            query.as_bytes(),
        ])
    ))
}

fn key_from_parts(directory: &Path, parts: &[&[u8]]) -> PathBuf {
    directory.join(format!("{}.bin", cache_key(parts.try_into().unwrap())))
}

#[test]
fn query_key_tracks_only_semantic_inputs() {
    let dir = TestDir::new();
    let key = key_path(dir.path(), MINILM_ID, false, "same query");
    assert_eq!(key, key_path(dir.path(), MINILM_ID, false, "same query"));
    assert_ne!(key, key_path(dir.path(), MINILM_ID, true, "same query"));
    assert_ne!(
        key,
        key_path(dir.path(), MINILM_ID, false, "different query")
    );
    assert_ne!(
        key,
        key_path(dir.path(), "other-model", false, "same query")
    );
    let base = key_from_parts(
        dir.path(),
        &[
            b"query-vectors-v2",
            b"model",
            b"sha",
            b"version",
            b"\0",
            b"query",
        ],
    );
    for variant in [
        key_from_parts(
            dir.path(),
            &[
                b"other-format",
                b"model",
                b"sha",
                b"version",
                b"\0",
                b"query",
            ],
        ),
        key_from_parts(
            dir.path(),
            &[
                b"query-vectors-v2",
                b"model",
                b"other-sha",
                b"version",
                b"\0",
                b"query",
            ],
        ),
        key_from_parts(
            dir.path(),
            &[
                b"query-vectors-v2",
                b"model",
                b"sha",
                b"other-version",
                b"\0",
                b"query",
            ],
        ),
    ] {
        assert_ne!(base, variant);
    }
    assert_ne!(
        key_from_parts(
            dir.path(),
            &[b"ab", b"c", b"sha", b"version", b"\0", b"query"]
        ),
        key_from_parts(
            dir.path(),
            &[b"a", b"bc", b"sha", b"version", b"\0", b"query"]
        )
    );
}

#[test]
fn cache_hit_preserves_exact_float_bits_and_skips_factory() {
    let dir = TestDir::new();
    let path = key_path(dir.path(), MINILM_ID, false, "query");
    let mut vector = unit_vector();
    vector[1] = f32::from_bits(0x8000_0000);
    assert!(cache_put(&path, &vector, |from, to| fs::rename(from, to)).is_some());
    let calls = Arc::new(AtomicUsize::new(0));
    let got = cached_query(
        Some(path),
        {
            let calls = Arc::clone(&calls);
            move || {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(unit_vector())
            }
        },
        || Ok(true),
    )
    .unwrap();
    assert_eq!(
        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        vector.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn gpu_and_skip_check_bypass_cache() {
    let dir = TestDir::new();
    let path = key_path(dir.path(), MINILM_ID, false, "query");
    let mut cached = vec![0.0; 384];
    cached[1] = 1.0;
    fs::write(&path, encode(&cached)).unwrap();
    let blocked = dir.path().join("not-a-directory");
    fs::write(&blocked, b"sentinel").unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let gpu = cached_query(None, || Ok(unit_vector()), {
        let calls = Arc::clone(&calls);
        move || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(true)
        }
    })
    .unwrap();
    assert_eq!(gpu, unit_vector());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let skip_calls = Arc::new(AtomicUsize::new(0));
    let skipped = cached_query(Some(path.clone()), || Ok(unit_vector()), {
        let calls = Arc::clone(&skip_calls);
        move || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(false)
        }
    })
    .unwrap();
    assert_eq!(skipped, unit_vector());
    assert_eq!(fs::read(blocked).unwrap(), b"sentinel");
    assert_eq!(fs::read(path).unwrap(), encode(&cached));
    assert_eq!(skip_calls.load(Ordering::SeqCst), 2);
}

#[test]
fn raw_cache_rejects_bad_length_nan_infinity_and_norm() {
    let dir = TestDir::new();
    let valid_path = key_path(dir.path(), MINILM_ID, false, "valid");
    let valid = unit_vector();
    fs::write(&valid_path, encode(&valid)).unwrap();
    assert_eq!(cache_get(&valid_path), Some(valid.try_into().unwrap()));
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
        let path = key_path(dir.path(), MINILM_ID, false, &i.to_string());
        fs::write(&path, bytes).unwrap();
        assert!(cache_get(&path).is_none());
    }
}

#[test]
fn failed_integrity_never_populates_cache() {
    let dir = TestDir::new();
    let path = key_path(dir.path(), MINILM_ID, false, "query");
    assert_eq!(
        cached_query(Some(path.clone()), || Ok(unit_vector()), || Ok(false)).unwrap(),
        unit_vector()
    );
    assert!(!path.exists());
}

#[test]
fn failed_integrity_error_propagates_without_cache_write() {
    let dir = TestDir::new();
    let path = key_path(dir.path(), MINILM_ID, false, "query");
    let error = cached_query(
        Some(path.clone()),
        || Ok(unit_vector()),
        || Err(anyhow::anyhow!("integrity failed")),
    )
    .unwrap_err();
    assert!(error.to_string().contains("integrity failed"));
    assert!(!path.exists());
}

#[test]
fn post_factory_integrity_recheck_prevents_store() {
    let dir = TestDir::new();
    let path = key_path(dir.path(), MINILM_ID, false, "query");
    let checks = Arc::new(AtomicUsize::new(0));
    assert_eq!(
        cached_query(Some(path.clone()), || Ok(unit_vector()), {
            let checks = Arc::clone(&checks);
            move || Ok(checks.fetch_add(1, Ordering::SeqCst) == 0)
        })
        .unwrap(),
        unit_vector()
    );
    assert!(!path.exists());
}

#[test]
fn concurrent_same_key_writes_remain_valid() {
    let dir = TestDir::new();
    let path = key_path(dir.path(), MINILM_ID, false, "query");
    let expected = encode(&unit_vector());
    fs::write(&path, &expected).unwrap();
    let (publishing, active) = std::sync::mpsc::channel();
    let (observed, read_done) = std::sync::mpsc::channel();
    let writer_path = path.clone();
    let mut writer_expected = vec![0.0; 384];
    writer_expected[1] = 1.0;
    let published = writer_expected.clone();
    let writer = std::thread::spawn(move || {
        assert!(
            cache_put(&writer_path, &writer_expected, |temp, destination| {
                publishing.send(()).unwrap();
                read_done.recv().unwrap();
                fs::rename(temp, destination)
            })
            .is_some()
        );
    });
    active.recv().unwrap();
    let mut bytes = [0; VECTOR_BYTES];
    File::open(&path).unwrap().read_exact(&mut bytes).unwrap();
    assert_eq!(bytes.len(), VECTOR_BYTES);
    assert_eq!(bytes, expected.as_slice());
    assert_eq!(cache_get(&path), Some(unit_vector().try_into().unwrap()));
    observed.send(()).unwrap();
    writer.join().unwrap();
    assert_eq!(
        cache_get(&path),
        Some(<[f32; 384]>::try_from(published).unwrap())
    );
    let writers: Vec<_> = (0..8)
        .map(|_| {
            let path = path.clone();
            std::thread::spawn(move || {
                for _ in 0..20 {
                    assert!(
                        cache_put(&path, &unit_vector(), |from, to| fs::rename(from, to)).is_some()
                    );
                }
            })
        })
        .collect();
    for writer in writers {
        writer.join().unwrap();
    }
    assert_eq!(cache_get(&path), Some(unit_vector().try_into().unwrap()));
}

#[test]
fn concurrent_distinct_key_stores_converge_to_file_bound() {
    let dir = TestDir::new();
    let workers: Vec<_> = (0..8)
        .map(|worker| {
            let directory = dir.path().to_path_buf();
            std::thread::spawn(move || {
                for index in 0..40 {
                    let path = key_path(&directory, MINILM_ID, false, &format!("{worker}-{index}"));
                    assert!(
                        cache_put(&path, &unit_vector(), |from, to| fs::rename(from, to)).is_some()
                    );
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
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "bin"))
        .count();
    assert!(count > 0 && count <= MAX_FILES);
}

#[test]
fn successful_store_prunes_to_exact_file_limit() {
    let dir = TestDir::new();
    for index in 0..=MAX_FILES {
        fs::write(dir.path().join(format!("old-{index:03}.bin")), b"old").unwrap();
    }
    let path = key_path(dir.path(), MINILM_ID, false, "new entry");
    assert!(cache_put(&path, &unit_vector(), |from, to| fs::rename(from, to)).is_some());
    assert_eq!(
        fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "bin"))
            .count(),
        MAX_FILES
    );
}

#[test]
fn read_only_cache_directory_preserves_search() {
    let dir = TestDir::new();
    let blocked = dir.path().join("file");
    fs::write(&blocked, b"still here").unwrap();
    let blocked_path = blocked.join("query.bin");
    assert_eq!(
        cached_query(Some(blocked_path), || Ok(unit_vector()), || Ok(true)).unwrap(),
        unit_vector()
    );
    assert_eq!(fs::read(blocked).unwrap(), b"still here");
}

#[test]
fn successful_publish_preserves_reacquired_temp_slot() {
    let dir = TestDir::new();
    let path = key_path(dir.path(), MINILM_ID, false, "query");
    let mut reacquired = None;
    assert!(cache_put(&path, &unit_vector(), |temp, destination| {
        fs::rename(temp, destination)?;
        let mut foreign = OpenOptions::new().write(true).create_new(true).open(temp)?;
        foreign.write_all(b"another writer's new slot")?;
        reacquired = Some(temp.to_path_buf());
        Ok(())
    })
    .is_some());
    assert_eq!(
        fs::read(reacquired.unwrap()).unwrap(),
        b"another writer's new slot"
    );
    assert_eq!(cache_get(&path), Some(unit_vector().try_into().unwrap()));
}

#[test]
fn rename_error_with_valid_winner_preserves_vector() {
    let dir = TestDir::new();
    let path = key_path(dir.path(), MINILM_ID, false, "query");
    let expected = unit_vector();
    let foreign = dir.path().join(".tmp-slot-31");
    fs::write(&foreign, b"another writer's slot").unwrap();
    let mut owned_temp = None;
    assert!(cache_put(&path, &expected, |temp, destination| {
        owned_temp = Some(temp.to_path_buf());
        fs::write(destination, encode(&expected))?;
        Err(std::io::Error::other(
            "simulated rename failure after winner publication",
        ))
    })
    .is_some());
    assert!(!owned_temp.unwrap().exists());
    assert_eq!(fs::read(foreign).unwrap(), b"another writer's slot");
    assert_eq!(cache_get(&path), Some(unit_vector().try_into().unwrap()));
}

#[test]
fn rename_error_with_corrupt_destination_never_unlinks_foreign_file() {
    let dir = TestDir::new();
    let path = key_path(dir.path(), MINILM_ID, false, "query");
    fs::write(&path, b"foreign corrupt entry").unwrap();
    let foreign = dir.path().join(".tmp-slot-31");
    fs::write(&foreign, b"owned by another writer").unwrap();
    assert!(
        cache_put(&path, &unit_vector(), |_owned_temp, _destination| Err(
            std::io::Error::other("simulated rename failure")
        ))
        .is_none()
    );
    assert_eq!(fs::read(&path).unwrap(), b"foreign corrupt entry");
    assert!(!dir.path().join(".tmp-slot-00").exists());
    assert_eq!(fs::read(&foreign).unwrap(), b"owned by another writer");
    assert!(cache_put(&path, &unit_vector(), |from, to| fs::rename(from, to)).is_some());
    assert_eq!(cache_get(&path), Some(unit_vector().try_into().unwrap()));
}

#[test]
fn abandoned_temp_slots_have_a_fixed_bound() {
    let dir = TestDir::new();
    let slot_bytes = b"abandoned slot remains byte-identical";
    for slot in 0..MAX_TEMP_FILES {
        fs::write(dir.path().join(format!(".tmp-slot-{slot:02}")), slot_bytes).unwrap();
    }
    let path = key_path(dir.path(), MINILM_ID, false, "uncached query");
    assert_eq!(
        cached_query(Some(path.clone()), || Ok(unit_vector()), || Ok(true)).unwrap(),
        unit_vector()
    );
    assert!(!dir.path().join(".tmp-slot-32").exists());
    for slot in 0..MAX_TEMP_FILES {
        assert_eq!(
            fs::read(dir.path().join(format!(".tmp-slot-{slot:02}"))).unwrap(),
            slot_bytes
        );
    }
    assert!(!path.exists());
    assert_eq!(
        fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "bin"))
            .count(),
        0
    );
}

#[test]
fn unknown_model_dispatch_preserves_factory_error_context() {
    let error = query_vector("unknown-model-for-query-cache-test", false, "q").unwrap_err();
    let chain = format!("{error:#}");
    assert!(chain.contains("load embedding model"));
    assert!(chain.contains("unknown embedder"));
}
