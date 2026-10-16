use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use sha2::Digest;

use super::MINILM_ID;

const MAX_FILES: usize = 256;
const MAX_TEMP_FILES: usize = 32;
const VECTOR_BYTES: usize = 384 * 4;
const FORMAT_TAG: &[u8] = b"query-vectors-v2";

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
    for (value, chunk) in vector.iter_mut().zip(bytes.chunks_exact(4)) {
        *value = f32::from_le_bytes(chunk.try_into().ok()?);
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
    let mut owned = None;
    for slot in 0..MAX_TEMP_FILES {
        let temp = directory.join(format!(".tmp-slot-{slot:02}"));
        match OpenOptions::new().write(true).create_new(true).open(&temp) {
            Ok(file) => {
                owned = Some((temp, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(_) => return None,
        }
    }
    let (temp, mut file) = owned?;
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
    if std::env::var_os("VEX_EMBEDDER_SKIP_CHECK").as_deref() == Some(std::ffi::OsStr::new("1")) {
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
    if files.len() > MAX_FILES {
        files.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let excess = files.len().saturating_sub(MAX_FILES);
        for (_, _, path) in files.into_iter().take(excess) {
            let _ = fs::remove_file(path);
        }
    }
}
