//! Bounded raw-vector cache for repeated semantic queries.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::Result;
use sha2::Digest;

use super::{Embedder, MINILM_ID};

const MAX_FILES: usize = 256;
const MAX_TEMP_FILES: usize = 32;
const VECTOR_BYTES: usize = 384 * 4;
const FORMAT_TAG: &[u8] = b"query-vectors-v2";

#[derive(Clone)]
struct QueryKey(String);

impl QueryKey {
    fn new(model_id: &str, normalized: bool, query: &str) -> Self {
        Self::from_parts(
            FORMAT_TAG,
            model_id.as_bytes(),
            super::integrity::MINILM_ONNX_SHA256.as_bytes(),
            env!("CARGO_PKG_VERSION").as_bytes(),
            normalized,
            query.as_bytes(),
        )
    }

    fn from_parts(
        format: &[u8],
        model_id: &[u8],
        onnx_sha256: &[u8],
        version: &[u8],
        normalized: bool,
        query_utf8: &[u8],
    ) -> Self {
        let mut digest = sha2::Sha256::new();
        let normalized = [u8::from(normalized)];
        for part in [
            format,
            model_id,
            onnx_sha256,
            version,
            normalized.as_slice(),
            query_utf8,
        ] {
            digest.update((part.len() as u64).to_le_bytes());
            digest.update(part);
        }
        Self(format!("{:x}.bin", digest.finalize()))
    }

    fn path(&self, directory: &Path) -> PathBuf {
        directory.join(&self.0)
    }
}

struct RawQueryCache {
    directory: PathBuf,
}

impl RawQueryCache {
    fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
        }
    }

    fn get(&self, key: &QueryKey) -> Option<[f32; 384]> {
        let mut bytes = Vec::with_capacity(VECTOR_BYTES + 1);
        File::open(key.path(&self.directory))
            .ok()?
            .take((VECTOR_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .ok()?;
        decode(&bytes)
    }

    fn put(&self, key: &QueryKey, vector: &[f32]) -> bool {
        self.put_with(key, vector, |from, to| std::fs::rename(from, to))
    }

    fn put_with<F>(&self, key: &QueryKey, vector: &[f32], publish: F) -> bool
    where
        F: FnOnce(&Path, &Path) -> std::io::Result<()>,
    {
        let Some(bytes) = encode_vector(vector) else {
            return false;
        };
        if fs::create_dir_all(&self.directory).is_err() {
            return false;
        }
        let mut owned = None;
        for slot in 0..MAX_TEMP_FILES {
            let path = self.directory.join(format!(".tmp-slot-{slot:02}"));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => {
                    owned = Some((path, file));
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(_) => return false,
            }
        }
        let Some((temp, mut file)) = owned else {
            return false;
        };
        if file.write_all(&bytes).is_err() {
            drop(file);
            let _ = fs::remove_file(&temp);
            return false;
        }
        drop(file);
        let destination = key.path(&self.directory);
        let stored = if publish(&temp, &destination).is_ok() {
            true
        } else {
            let winner = self.get(key).is_some();
            let _ = fs::remove_file(&temp);
            winner
        };
        if stored {
            prune(&self.directory);
        }
        stored
    }
}

pub(crate) fn make_query_embedder(id: &str, normalized: bool) -> Result<Box<dyn Embedder>> {
    let factory_id = id.to_owned();
    make_query_embedder_with(
        id,
        normalized,
        true,
        RawQueryCache::new(query_cache_directory()),
        move || super::make_embedder(&factory_id),
        query_integrity_eligible,
    )
}

fn query_cache_directory() -> PathBuf {
    crate::util::config::embed_cache_dir().join("query-vectors-v2")
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

fn make_query_embedder_with<F, E>(
    id: &str,
    normalized: bool,
    cpu: bool,
    cache: RawQueryCache,
    mut factory: F,
    eligible: E,
) -> Result<Box<dyn Embedder>>
where
    F: FnMut() -> Result<Box<dyn Embedder>> + Send + 'static,
    E: Fn() -> Result<bool> + Send + 'static,
{
    if id != MINILM_ID || !cpu {
        return factory();
    }
    Ok(Box::new(QueryEmbedder {
        normalized,
        cache,
        factory,
        eligible,
        inner: None,
    }))
}

struct QueryEmbedder<F, E> {
    normalized: bool,
    cache: RawQueryCache,
    factory: F,
    eligible: E,
    inner: Option<Box<dyn Embedder>>,
}

impl<F, E> Embedder for QueryEmbedder<F, E>
where
    F: FnMut() -> Result<Box<dyn Embedder>> + Send,
    E: Fn() -> Result<bool> + Send,
{
    fn id(&self) -> &'static str {
        MINILM_ID
    }
    fn dim(&self) -> u32 {
        384
    }
    fn char_budget(&self) -> usize {
        super::MINILM_CHAR_BUDGET
    }
    fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let key = QueryKey::new(MINILM_ID, self.normalized, text);
        if (self.eligible)()? {
            if let Some(value) = self.cache.get(&key) {
                tracing::debug!("query vector cache: hit before factory");
                return Ok(value.to_vec());
            }
        }
        if self.inner.is_none() {
            self.inner = Some((self.factory)()?);
        }
        let vector = self
            .inner
            .as_mut()
            .expect("initialized above")
            .embed(text)?;
        if (self.eligible)()? {
            self.cache.put(&key, &vector);
        }
        Ok(vector)
    }
    fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|text| self.embed(text)).collect()
    }
}

#[cfg(test)]
#[path = "query_cache/tests.rs"]
mod tests;

fn decode(bytes: &[u8]) -> Option<[f32; 384]> {
    if bytes.len() != VECTOR_BYTES {
        return None;
    }
    let mut vector = [0.0; 384];
    for (value, chunk) in vector.iter_mut().zip(bytes.chunks_exact(4)) {
        *value = f32::from_le_bytes(chunk.try_into().ok()?);
    }
    valid(&vector).then_some(vector)
}

fn encode_vector(vector: &[f32]) -> Option<Vec<u8>> {
    if vector.len() != 384 || !valid(vector) {
        return None;
    }
    Some(
        vector
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
}
fn valid(vector: &[f32]) -> bool {
    vector.iter().all(|value| value.is_finite())
        && ((vector
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            - 1.0)
            .abs()
            <= 0.001)
}

fn prune(directory: &Path) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    let mut files: Vec<_> = entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let path = entry.path();
            if !path.extension().is_some_and(|ext| ext == "bin")
                || !entry.file_type().ok()?.is_file()
            {
                return None;
            }
            Some((
                entry.metadata().ok()?.modified().ok()?,
                entry.file_name(),
                path,
            ))
        })
        .collect();
    if files.len() <= MAX_FILES {
        return;
    }
    files.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let excess = files.len().saturating_sub(MAX_FILES);
    for (_, _, path) in files.into_iter().take(excess) {
        let _ = fs::remove_file(path);
    }
}
