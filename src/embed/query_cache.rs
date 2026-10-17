use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use sha2::Digest;

use super::MINILM_ID;

const MAX_FILES: usize = 256;
const TEMP_MAX_AGE: Duration = Duration::from_secs(60);
const VECTOR_BYTES: usize = 384 * 4;
const FORMAT_TAG: &[u8] = b"query-vectors-v2";
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

fn cache_key(parts: [&[u8]; 6]) -> String {
    let mut digest = sha2::Sha256::new();
    for part in parts {
        digest.update((part.len() as u64).to_le_bytes());
        digest.update(part);
    }
    format!("{:x}", digest.finalize())
}

fn cache_get(path: &Path) -> Option<[f32; 384]> {
    let mut file = File::open(path).ok()?;
    let mut bytes = [0; VECTOR_BYTES];
    file.read_exact(&mut bytes).ok()?;
    let mut extra = [0];
    if file.read(&mut extra).ok()? != 0 {
        return None;
    }
    let mut vector = [0.0; 384];
    for (value, chunk) in vector.iter_mut().zip(bytes.as_chunks::<4>().0) {
        *value = f32::from_le_bytes(*chunk);
    }
    valid(&vector).then_some(vector)
}

fn cache_put(
    path: &Path,
    vector: &[f32],
    publish: impl FnOnce(&Path, &Path) -> std::io::Result<()>,
) -> Option<()> {
    if vector.len() != 384 || !valid(vector) {
        return None;
    }
    let directory = path.parent()?;
    fs::create_dir_all(directory).ok()?;
    let (temp, mut file) = create_temp(directory).ok()?;
    let bytes: Vec<_> = vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let wrote = file.write_all(&bytes).is_ok();
    drop(file);
    let published = wrote && publish(&temp, path).is_ok();
    let stored = published || (wrote && cache_get(path).is_some());
    if !published {
        let _ = fs::remove_file(&temp);
    }
    stored.then(|| prune(directory))
}

pub(crate) fn query_vector(id: &str, normalized: bool, query: &str) -> Result<Vec<f32>> {
    if id != MINILM_ID {
        return super::make_embedder(id)
            .context("load embedding model")?
            .embed(query)
            .context("embed query");
    }
    let directory = crate::util::config::embed_cache_dir().join("query-vectors-v2");
    let normalized = [u8::from(normalized)];
    // make_embedder is CPU-only; if this ever routes through make_embedder_with_device,
    // the resolved device must become part of the cache key to avoid mixing vectors.
    let key = cache_key([
        FORMAT_TAG,
        id.as_bytes(),
        super::integrity::MINILM_ONNX_SHA256.as_bytes(),
        env!("CARGO_PKG_VERSION").as_bytes(),
        &normalized,
        query.as_bytes(),
    ]);
    cached_query(
        Some(directory.join(format!("{key}.bin"))),
        || super::make_embedder(id)?.embed(query),
        query_integrity_eligible,
    )
    .context("embed query")
}

fn query_integrity_eligible() -> Result<bool> {
    if super::integrity::skip_check() {
        return Ok(false);
    }
    let cache = crate::util::config::embed_cache_dir();
    let Some(path) = super::integrity::find_minilm_onnx(&cache) else {
        return Ok(false);
    };
    super::integrity::verify_with_marker(&path, super::integrity::MINILM_ONNX_SHA256)?;
    Ok(true)
}

fn cached_query(
    path: Option<PathBuf>,
    factory: impl FnOnce() -> Result<Vec<f32>>,
    eligible: impl Fn() -> Result<bool>,
) -> Result<Vec<f32>> {
    let Some(path) = path else {
        return factory();
    };
    if eligible()? {
        if let Some(vector) = cache_get(&path) {
            tracing::debug!("query vector cache: hit before factory");
            return Ok(vector.to_vec());
        }
    }
    let vector = factory()?;
    if eligible()? {
        let _ = cache_put(&path, &vector, |from, to| fs::rename(from, to));
    }
    Ok(vector)
}

#[cfg(test)]
#[path = "query_cache/tests.rs"]
mod tests;

fn valid(vector: &[f32]) -> bool {
    let norm: f64 = vector.iter().map(|&v| f64::from(v).powi(2)).sum();
    norm.is_finite() && (norm - 1.0).abs() <= 0.001
}

fn prune(directory: &Path) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    let mut files: Vec<_> = entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != "bin") || !entry.file_type().ok()?.is_file()
            {
                return None;
            }
            let modified = entry.metadata().ok()?.modified().ok()?;
            Some((modified, entry.file_name(), path))
        })
        .collect();
    let now = SystemTime::now();
    for entry in fs::read_dir(directory).into_iter().flatten().flatten() {
        let path = entry.path();
        if !entry.file_type().is_ok_and(|kind| kind.is_file())
            || !entry.file_name().to_string_lossy().starts_with(".tmp")
        {
            continue;
        }
        let Ok(modified) = entry.metadata().and_then(|metadata| metadata.modified()) else {
            continue;
        };
        if now
            .duration_since(modified)
            .is_ok_and(|age| age >= TEMP_MAX_AGE)
        {
            let _ = fs::remove_file(path);
        }
    }
    if files.len() > MAX_FILES {
        files.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let excess = files.len().saturating_sub(MAX_FILES);
        for (_, _, path) in files.into_iter().take(excess) {
            let _ = fs::remove_file(path);
        }
    }
}

fn create_temp(directory: &Path) -> std::io::Result<(PathBuf, File)> {
    loop {
        let nonce = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let temp = directory.join(format!(
            ".tmp-query-{}-{timestamp}-{nonce}",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&temp) {
            Ok(file) => return Ok((temp, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
}
