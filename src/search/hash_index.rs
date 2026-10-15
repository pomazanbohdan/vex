//! v1.14.1 B1.1 — HNSW hash-index sidecar.
//!
//! Persists `Vec<u64>` of `context_hash` values in sym_idx order so the
//! semantic-search query path can map HNSW results (keyed by content
//! hash, not by `sym_idx`) back to `SymbolRecord` positions. The hash
//! keying is the prerequisite for B1.2 incremental update: a symbol's
//! key is stable across `vex update` runs (content-based), while the
//! old `sym_idx`-as-key scheme broke whenever any earlier file's
//! symbol count changed.
//!
//! Lives at `<index_dir>/index.hashes` next to `index.hnsw` and is
//! always co-written + co-removed with it. Stale or missing sidecar
//! makes `HnswHandle::open` bail to brute-force search, same
//! degradation path as a stale HNSW file itself.
//!
//! ## Format (binary, little-endian)
//!
//! ```text
//! magic:        4 bytes "VEXH"
//! version:      u32 = 1 (public/unbound) or 2 (private/bound)
//! count:        u32 (number of hashes)
//! v2 digest:    u64 (ordered symbol-mapping evidence)
//! data:         count × u64 (sym_idx-ordered; follows digest in v2)
//! ```
//!
//! No `embedder_id` field — the embed cache (`embed/cache.rs`) already
//! gates that boundary, and the sidecar is paired with `index.hnsw`
//! which carries the embedder's dimension in its own header. Magic /
//! version mismatch on load → discard, fall back to brute force.

use std::io::{BufReader, Read, Write};
use std::path::Path;

use crate::store::reader::IndexReader;
use anyhow::{Context, Result};
use xxhash_rust::xxh3::Xxh3Default;

const MAGIC: &[u8; 4] = b"VEXH";
const VERSION_V1: u32 = 1;
const VERSION_V2: u32 = 2;

/// Bound on the count field. ~10M symbols is past anything legitimate
/// (a 384-dim f32 vector blob alone would be 15 GB at that scale). The
/// reader bails before allocating against a crafted count.
const MAX_COUNT: u32 = 10_000_000;

pub(crate) fn matches_hnsw<'a>(
    index: &usearch::Index,
    expected_dim: usize,
    unique_hashes: impl ExactSizeIterator<Item = &'a u64>,
) -> bool {
    index.dimensions() == expected_dim
        && matches!(index.metric_kind(), usearch::MetricKind::Cos)
        && index.size() > 0
        && index.size() == unique_hashes.len()
        && unique_hashes
            .into_iter()
            .all(|&hash| index.count(hash) == 1)
}

/// Atomic save: write to `.tmp`, then rename. Matches the convention
/// used by `embed::cache::EmbedCache::save` and `store::writer` —
/// half-written sidecars can't be mistaken for valid ones.
///
/// **Note (v1.15.0 C):** the production hot path
/// (`pipeline::output::commit_hnsw_and_sidecar`) no longer calls this
/// wrapper — it uses [`save_to_tmp`] directly so the HNSW and sidecar
/// renames can be batched two-phase. `save` is retained as the
/// convenience entry point for any future caller that wants a single-
/// file atomic write without the two-phase orchestration. Unit tests
/// in this module exercise both `save` and the split form.
#[allow(dead_code)]
pub fn save(path: &Path, hashes: &[u64]) -> Result<()> {
    let tmp = path.with_extension("hashes.tmp");
    save_to_tmp(&tmp, hashes)?;
    if let Err(e) = std::fs::rename(&tmp, path) {
        // Best-effort cleanup if the rename failed (partial cross-FS
        // moves on Linux can leave the tmp behind).
        let _ = std::fs::remove_file(&tmp);
        return Err(e).with_context(|| format!("rename {} → {}", tmp.display(), path.display()));
    }
    Ok(())
}

/// v1.15.0 C — split out the "write tmp" half of [`save`] so the
/// orchestrator in `pipeline::output` can do a two-phase atomic
/// commit of the HNSW + sidecar pair: write both `.tmp` files,
/// fsync, then rename both back-to-back. The inconsistency window
/// (one renamed, one not) shrinks from "between two file-content
/// writes" (~ms on a large sidecar) to "between two rename syscalls"
/// (~μs) — small enough that `HnswHandle::open`'s size-check
/// fallback rarely fires in practice.
///
/// `tmp_path` is the destination of the tmp write; the caller picks
/// the path (typically `final_path.with_extension("hashes.tmp")`).
/// On success the tmp file exists, is fsync'd, and is ready to be
/// renamed to the final path via plain `std::fs::rename`. On
/// failure the tmp file may exist in a partial state; callers
/// should treat any error as a "tmp leak" they need to clean up.
pub fn save_to_tmp(tmp_path: &Path, hashes: &[u64]) -> Result<()> {
    write_sidecar(tmp_path, hashes, None)
}

/// Write a private v2 sidecar bound to the ordered symbols in `reader`.
pub(crate) fn save_bound_to_tmp(
    tmp_path: &Path,
    hashes: &[u64],
    reader: &IndexReader,
) -> Result<()> {
    let digest = mapping_digest(reader, hashes)?;
    write_sidecar(tmp_path, hashes, Some(digest))
}

fn write_sidecar(tmp_path: &Path, hashes: &[u64], digest: Option<u64>) -> Result<()> {
    let version = if digest.is_some() {
        VERSION_V2
    } else {
        VERSION_V1
    };
    // Match the read-side `MAX_COUNT` guard so a truncating `as u32` cast
    // can't silently write a wrong count for an index past 10M symbols —
    // the cast would otherwise produce a sidecar that `load` accepts as
    // valid but is short by `hashes.len() % (MAX_COUNT + 1)` entries.
    anyhow::ensure!(
        hashes.len() <= MAX_COUNT as usize,
        "hash count exceeds MAX_COUNT: {} > {}",
        hashes.len(),
        MAX_COUNT,
    );
    let mut file = std::fs::File::create(tmp_path)
        .with_context(|| format!("create {}", tmp_path.display()))?;
    file.write_all(MAGIC).context("write magic")?;
    file.write_all(&version.to_le_bytes())
        .context("write version")?;
    file.write_all(&(hashes.len() as u32).to_le_bytes())
        .context("write count")?;
    if let Some(digest) = digest {
        file.write_all(&digest.to_le_bytes())
            .context("write symbol-mapping digest")?;
    }
    for h in hashes {
        file.write_all(&h.to_le_bytes()).context("write hash")?;
    }
    // fsync so the bytes are durable on disk before the caller's
    // rename — matches the `store::writer` convention (see comment
    // at writer.rs:732 — "without sync_all() between flush and
    // rename, a crash can leave the renamed path pointing at an
    // empty file").
    file.sync_all().context("fsync hash-index tmp")?;
    Ok(())
}

/// Load and validate. Returns an error on any failure (file missing,
/// magic mismatch, version mismatch, absurd count, truncated body) so
/// the caller can decide whether to bail or fall back. Callers that
/// want "absence ≡ no sidecar" semantics should check `path.exists()`
/// first.
pub fn load(path: &Path) -> Result<Vec<u64>> {
    read_sidecar(path).map(|(_, hashes, _)| hashes)
}

pub(crate) fn load_bound(path: &Path, reader: &IndexReader) -> Result<Vec<u64>> {
    let (version, hashes, digest) = read_sidecar(path)?;
    anyhow::ensure!(
        version == VERSION_V2,
        "hash-index mapping evidence is absent; re-run `vex index --semantic`"
    );
    let expected = mapping_digest(reader, &hashes)?;
    anyhow::ensure!(
        digest == Some(expected),
        "hash-index symbol mapping mismatch; re-run `vex index --semantic`"
    );
    Ok(hashes)
}

fn read_sidecar(path: &Path) -> Result<(u32, Vec<u64>, Option<u64>)> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut file = BufReader::new(file);

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).context("read magic")?;
    if &magic != MAGIC {
        anyhow::bail!("hash-index magic mismatch (got {:?})", magic);
    }

    let version = read_u32(&mut file).context("read version")?;
    if version != VERSION_V1 && version != VERSION_V2 {
        anyhow::bail!("hash-index version mismatch: {} not in [1, 2]", version);
    }

    let count = read_u32(&mut file).context("read count")?;
    if count > MAX_COUNT {
        anyhow::bail!("hash-index count absurd: {} > {}", count, MAX_COUNT);
    }

    let digest = if version == VERSION_V2 {
        Some(read_u64(&mut file).context("read symbol-mapping digest")?)
    } else {
        None
    };

    let mut hashes: Vec<u64> = Vec::with_capacity(count as usize);
    for _ in 0..count {
        hashes.push(read_u64(&mut file).context("read hash")?);
    }
    Ok((version, hashes, digest))
}

fn mapping_digest(reader: &IndexReader, hashes: &[u64]) -> Result<u64> {
    anyhow::ensure!(
        hashes.len() == reader.symbol_count(),
        "hash-index count {} disagrees with index symbol count {}",
        hashes.len(),
        reader.symbol_count()
    );
    let mut hasher = Xxh3Default::new();
    hasher.update(&(hashes.len() as u64).to_le_bytes());
    hasher.update(&reader.header().vector_dim.to_le_bytes());
    for (idx, &hash) in hashes.iter().enumerate() {
        let record = reader
            .symbol(idx)
            .with_context(|| format!("missing symbol record {idx} while binding hash sidecar"))?;
        hasher.update(&hash.to_le_bytes());
        hasher.update(&[record.kind]);
        hasher.update(&record.line.to_le_bytes());
        hasher.update(&record.vector_index.to_le_bytes());
        update_string(&mut hasher, reader.read_string(record.file_offset));
        update_string(&mut hasher, reader.read_string(record.name_offset));
        update_string(&mut hasher, reader.read_string(record.signature_offset));
    }
    Ok(hasher.digest())
}

fn update_string(hasher: &mut Xxh3Default, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

/// Fuzzing shim — exposed only to `vex-fuzz`. Writes `data` as a
/// hash-index sidecar to a process-temp path, then drives [`load`].
/// Goal: no panics, no UB on any byte sequence the parser sees, since
/// the sidecar is loaded at every `vex search --semantic` query and
/// `HnswHandle::open` would otherwise propagate a panic to the caller.
/// Mirrors the [`crate::embed::integrity::__fuzz_marker_bytes`] pattern
/// (a `pub(crate)` parser exercised via a doc-hidden shim so the
/// public API stays narrow).
#[doc(hidden)]
pub fn __fuzz_hash_index_bytes(data: &[u8]) {
    let path = std::env::temp_dir().join("__vex_fuzz_hash_index.bin");
    // Tmp I/O failure (full disk / permission) — return rather than
    // assert; libfuzzer will move on with a new sample.
    if std::fs::write(&path, data).is_err() {
        return;
    }
    // The parser's contract is "any malformed input is an `Err`, never
    // a panic". We discard the result — libfuzzer's success metric is
    // process survival, not a returned value.
    let _ = load(&path);
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn round_trip_empty() {
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().join("h.bin");
        save(&p, &[]).unwrap();
        let back = load(&p).unwrap();
        assert!(back.is_empty());
    }

    #[test]
    fn round_trip_preserves_order_and_content() {
        // Sym_idx order MUST be preserved — the `HnswHandle::open`
        // builder relies on `hashes[i] -> sym_idx == i` to materialise
        // the hash→sym_idx map. A re-ordering bug here would silently
        // mis-attribute every semantic-search result.
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().join("h.bin");
        let hashes = vec![1, 2, 3, 0xDEADBEEF, u64::MAX];
        save(&p, &hashes).unwrap();
        let back = load(&p).unwrap();
        assert_eq!(back, hashes);
    }

    #[test]
    fn save_is_atomic_via_tmp_rename() {
        // Same atomic-write contract `embed::cache::EmbedCache::save`
        // follows — a crash mid-save can't produce a partially-written
        // sidecar that load() would treat as valid. Pin it.
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().join("h.bin");
        save(&p, &[1, 2, 3]).unwrap();
        let leftover = p.with_extension("hashes.tmp");
        assert!(!leftover.exists(), "tmp must be renamed away");
        assert!(p.exists(), "final file must exist");
    }

    #[test]
    fn rejects_bad_magic() {
        // Crafted sidecar with garbage magic — load must reject rather
        // than parse the rest as if it were our format.
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().join("h.bin");
        let mut bad = Vec::new();
        bad.extend_from_slice(b"NOPE");
        bad.extend_from_slice(&VERSION_V1.to_le_bytes());
        bad.extend_from_slice(&0u32.to_le_bytes());
        std::fs::write(&p, bad).unwrap();
        assert!(load(&p).is_err());
    }

    #[test]
    fn rejects_bad_version() {
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().join("h.bin");
        let mut bad = Vec::new();
        bad.extend_from_slice(MAGIC);
        bad.extend_from_slice(&999u32.to_le_bytes());
        bad.extend_from_slice(&0u32.to_le_bytes());
        std::fs::write(&p, bad).unwrap();
        assert!(load(&p).is_err());
    }

    #[test]
    fn rejects_truncated_v2_mapping_header() {
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().join("h.bin");
        let mut bad = Vec::new();
        bad.extend_from_slice(MAGIC);
        bad.extend_from_slice(&VERSION_V2.to_le_bytes());
        bad.extend_from_slice(&0u32.to_le_bytes());
        std::fs::write(&p, bad).unwrap();
        assert!(load(&p).is_err());
    }

    #[test]
    fn rejects_absurd_count() {
        // A crafted count past `MAX_COUNT` would allocate a huge Vec
        // before reading the body fails — bail at the header instead.
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().join("h.bin");
        let mut bad = Vec::new();
        bad.extend_from_slice(MAGIC);
        bad.extend_from_slice(&VERSION_V1.to_le_bytes());
        bad.extend_from_slice(&(MAX_COUNT + 1).to_le_bytes());
        std::fs::write(&p, bad).unwrap();
        assert!(load(&p).is_err());
    }

    #[test]
    fn rejects_truncated_body() {
        // Header claims 3 hashes, body has 1 — the read_u64 loop bails
        // on EOF and the function surfaces the error rather than
        // silently returning a short Vec.
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().join("h.bin");
        let mut bad = Vec::new();
        bad.extend_from_slice(MAGIC);
        bad.extend_from_slice(&VERSION_V1.to_le_bytes());
        bad.extend_from_slice(&3u32.to_le_bytes());
        bad.extend_from_slice(&42u64.to_le_bytes());
        std::fs::write(&p, bad).unwrap();
        assert!(load(&p).is_err());
    }

    #[test]
    fn missing_file_returns_err() {
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().join("nope.bin");
        // Caller is expected to check `exists()` first if it wants
        // "absence ≡ no sidecar" semantics; the loader doesn't decide
        // policy for them.
        assert!(load(&p).is_err());
    }
}
