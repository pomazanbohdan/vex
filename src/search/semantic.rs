use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

use crate::embed::Embedder;
use crate::index::symbols::SymbolKind;
use crate::search::{hash_index, MatchType, SearchResult};
use crate::store::reader::IndexReader;

/// Semantic search using HNSW index (fast) with brute-force fallback.
///
/// `normalized` reflects the manifest's `vectors_normalized` flag —
/// when `true`, the stored vectors are unit-length and the brute-force
/// fallback uses dot product instead of full cosine. The query vector
/// is normalized here so it matches.
pub fn search_with_embedder(
    reader: &IndexReader,
    embedder: &mut dyn Embedder,
    query: &str,
    top_k: usize,
    hnsw_path: &Path,
    normalized: bool,
) -> Result<Vec<SearchResult>> {
    let mut query_vec = embedder.embed(query).context("embed query")?;
    if normalized {
        normalize_in_place(&mut query_vec);
    }

    let scored = search_hnsw_at(hnsw_path, &query_vec, top_k, reader.symbol_count())
        .unwrap_or_else(|| search_brute_force(reader, &query_vec, top_k, normalized));

    let results = scored
        .into_iter()
        .filter_map(|(idx, score)| {
            let rec = reader.symbol(idx)?;
            let name = reader.read_string(rec.name_offset).to_string();
            let path = reader.read_string(rec.file_offset).to_string();
            let sig = {
                let s = reader.read_string(rec.signature_offset);
                if s.is_empty() {
                    None
                } else {
                    Some(s.to_string())
                }
            };

            Some(SearchResult {
                name,
                kind: SymbolKind::try_from(rec.kind)
                    .map_or("unknown", |k| k.as_str())
                    .to_string(),
                path,
                line: rec.line as usize,
                signature: sig,
                score: score as f64,
                match_type: MatchType::Semantic,
            })
        })
        .collect();

    Ok(results)
}

/// v1.13 P1: an opened HNSW index handle, reusable across many
/// queries. The previous `search_hnsw_at` created + `view()`-ed a
/// fresh `usearch::Index` per call — `find_duplicates`'s outer loop
/// then mmap'd the HNSW file `symbol_count` times per invocation. The
/// handle lifts that cost to once per query.
///
pub(crate) struct HnswHandle {
    index: usearch::Index,
    hash_to_sym_indices: HashMap<u64, Vec<u32>>,
}

impl HnswHandle {
    /// Open the HNSW file at `hnsw_path` and prepare it for repeated
    /// `search_with_aliases()` calls. Returns `None` for the same reasons
    /// `search_hnsw_at` previously returned `None`, plus the new
    /// v1.14.1 B1.1 cases:
    /// - HNSW file missing
    /// - index creation / view failed
    /// - the persisted index is empty
    /// - the paired `index.hashes` sidecar is missing or malformed
    ///   (pre-1.14.1 indexes have no sidecar — re-run `vex index
    ///   --semantic` to upgrade)
    /// - the sidecar length disagrees with `expected_symbols`
    pub(crate) fn open(
        hnsw_path: &Path,
        query_dim: usize,
        expected_symbols: usize,
    ) -> Option<Self> {
        use usearch::{new_index, IndexOptions, MetricKind, ScalarKind};

        if !hnsw_path.exists() {
            return None;
        }

        let options = IndexOptions {
            dimensions: query_dim,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: 0,
            expansion_add: 0,
            expansion_search: 0,
            multi: false,
        };

        let index = new_index(&options)
            .map_err(|e| tracing::warn!(error = %e, "failed to create HNSW index handle"))
            .ok()?;

        let path_str = hnsw_path.to_str()?;
        index
            .view(path_str)
            .map_err(|e| {
                tracing::warn!(path = %hnsw_path.display(), error = %e, "failed to view HNSW index")
            })
            .ok()?;

        if index.size() == 0 {
            return None;
        }

        // Load the hash-index sidecar that pairs HNSW hash-keys with
        // sym_idx positions. Sibling file `index.hashes`; if missing or
        // malformed the HNSW is unusable (we'd return correct top-k by
        // hash but couldn't map back to records[]) — bail to brute force.
        let hash_index_path = hnsw_path.parent()?.join("index.hashes");
        let hashes = hash_index::load(&hash_index_path)
            .map_err(|e| {
                tracing::warn!(
                    path = %hash_index_path.display(),
                    error = %e,
                    "HNSW hash-index sidecar missing or malformed — falling back to brute-force"
                )
            })
            .ok()?;
        if hashes.len() != expected_symbols {
            tracing::warn!(
                sidecar = hashes.len(),
                symbols = expected_symbols,
                "HNSW hash-index sidecar size mismatch — falling back to brute-force"
            );
            return None;
        }

        let mut hash_to_sym_indices: HashMap<u64, Vec<u32>> = HashMap::with_capacity(hashes.len());
        for (i, h) in hashes.into_iter().enumerate() {
            hash_to_sym_indices.entry(h).or_default().push(i as u32);
        }
        if !hash_index::matches_hnsw(&index, query_dim, hash_to_sym_indices.keys()) {
            tracing::warn!(
                hnsw = index.size(),
                unique_hashes = hash_to_sym_indices.len(),
                dimensions = index.dimensions(),
                query_dim,
                "HNSW dimension/key-set mismatch — falling back to brute-force"
            );
            return None;
        }

        tracing::debug!(
            symbols = expected_symbols,
            unique_vectors = index.size(),
            aliases = expected_symbols - index.size(),
            "HNSW index opened"
        );

        Some(Self {
            index,
            hash_to_sym_indices,
        })
    }

    pub(crate) fn search_with_aliases(
        &self,
        query_vec: &[f32],
        distinct_hash_limit: usize,
    ) -> Option<Vec<(usize, f32)>> {
        if query_vec.len() != self.index.dimensions() {
            return None;
        }
        if distinct_hash_limit == 0 {
            return Some(Vec::new());
        }
        let results = self
            .index
            .search(query_vec, distinct_hash_limit.min(self.index.size()))
            .ok()?;
        let mut scored = Vec::new();
        for (&key, &dist) in results.keys.iter().zip(&results.distances) {
            let aliases = self.hash_to_sym_indices.get(&key)?;
            let similarity = (1.0 - dist).max(0.0);
            scored.extend(aliases.iter().map(|&idx| (idx as usize, similarity)));
        }
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        Some(scored)
    }
}

/// Single-call wrapper kept for the search-with-embedder path and for
/// any caller that does exactly one HNSW lookup per `vex` invocation.
/// New multi-query loops should open a [`HnswHandle`] once instead.
pub(crate) fn search_hnsw_at(
    hnsw_path: &Path,
    query_vec: &[f32],
    distinct_hash_limit: usize,
    expected_symbols: usize,
) -> Option<Vec<(usize, f32)>> {
    HnswHandle::open(hnsw_path, query_vec.len(), expected_symbols)?
        .search_with_aliases(query_vec, distinct_hash_limit)
}

/// Brute-force cosine similarity over all stored vectors. O(N).
///
/// `normalized` gates the v1.13 P5 fast path: when `true`, vectors are
/// already L2-normalized so `dot_product` is mathematically equivalent
/// to cosine and skips the per-call `sqrt` + norm computations.
fn search_brute_force(
    reader: &IndexReader,
    query_vec: &[f32],
    top_k: usize,
    normalized: bool,
) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = Vec::new();

    for i in 0..reader.symbol_count() {
        if let Some(rec) = reader.symbol(i) {
            if let Some(vec) = reader.vector(rec.vector_index) {
                let sim = if normalized {
                    dot_product(query_vec, vec)
                } else {
                    cosine_similarity(query_vec, vec)
                };
                scored.push((i, sim));
            }
        }
    }

    // Tie-break on symbol index so equal-similarity scores produce a
    // deterministic ordering across runs. Without this, two `vex search
    // --semantic` invocations on the same index can disagree at the
    // tie-break boundary.
    //
    // Note: `docs/RANKING-EVAL.md` specifies `(path, name, line)` as the
    // canonical total-order secondary key. This per-channel pre-fusion
    // sort uses `sym_idx` instead because (a) the surrounding `fuse_many`
    // re-applies the `(path, name, line)` order at the fusion layer where
    // results have a `SearchResult` shape, and (b) resolving the symbol
    // index back to a `SearchResult` here would require an extra reader
    // lookup per comparison. `sym_idx` is itself deterministic (writer
    // assembly order is fixed), so the contract is preserved end-to-end
    // even though this specific sort uses a coarser key.
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.truncate(top_k);
    scored
}

pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// v1.13 P5 fast path. Equivalent to [`cosine_similarity`] **only when
/// both inputs are L2-normalized**. Skips the per-call `sqrt` + norm
/// computations. Caller must guarantee the inputs are unit vectors;
/// the brute-force search loop gates on the manifest's
/// `vectors_normalized` flag.
#[inline]
pub(crate) fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2-normalize `v` in place. No-op on zero vectors (would otherwise
/// divide by zero — they remain all-zero, which `dot_product` will
/// score as 0.0, matching `cosine_similarity`'s short-circuit).
pub(crate) fn normalize_in_place(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alias_fixture() -> (tempfile::TempDir, std::path::PathBuf, Vec<Vec<f32>>) {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("index.hnsw");
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        crate::index::pipeline::build_hnsw_at(
            &path,
            &tmp.path().join("index.hashes"),
            &vectors,
            &[11, 11, 22],
        )
        .unwrap();
        (tmp, path, vectors)
    }

    #[test]
    fn hnsw_aliases_survive_save_load_and_expand_before_filtering() {
        let (_tmp, path, vectors) = alias_fixture();
        let handle = HnswHandle::open(&path, 3, 3).expect("duplicate contexts must use ANN");
        let scored = handle.search_with_aliases(&vectors[0], 1).unwrap();
        assert_eq!(scored.iter().map(|r| r.0).collect::<Vec<_>>(), [0, 1]);
        assert!(scored.iter().all(|r| r.1 > 0.99));
        assert!(handle.search_with_aliases(&[1.0, 0.0], 1).is_none());
        assert!(handle
            .search_with_aliases(&vectors[0], 0)
            .unwrap()
            .is_empty());

        let scope = crate::cli::scope::PathScope::from_args(&["included/**".into()], &[]).unwrap();
        let metadata = [("excluded/a.rs", "function"), ("included/b.rs", "method")];
        let scoped = scored
            .iter()
            .filter(|r| scope.accept(metadata[r.0].0) && metadata[r.0].1 == "method")
            .collect::<Vec<_>>();
        assert_eq!(scoped.len(), 1);
        assert_eq!(scoped[0].0, 1);
    }

    #[test]
    fn hnsw_rejects_same_size_stale_keys_and_wrong_dimension() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("index.hnsw");
        let hashes = tmp.path().join("index.hashes");
        crate::index::pipeline::build_hnsw_at(
            &path,
            &hashes,
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
            &[11, 22],
        )
        .unwrap();
        assert!(HnswHandle::open(&path, 3, 2).is_none());
        hash_index::save(&hashes, &[11, 33]).unwrap();
        assert!(HnswHandle::open(&path, 2, 2).is_none());
    }

    #[test]
    fn hnsw_aliases_remain_visible_in_semantic_similar_and_duplicates() {
        use crate::index::symbols::{ParsedFile, ParsedSymbol};

        struct Query;

        impl Embedder for Query {
            fn id(&self) -> &'static str {
                "fixture"
            }

            fn dim(&self) -> u32 {
                3
            }

            fn char_budget(&self) -> usize {
                256
            }

            fn embed(&mut self, _: &str) -> Result<Vec<f32>> {
                Ok(vec![1.0, 0.0, 0.0])
            }

            fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
                texts.iter().map(|text| self.embed(text)).collect()
            }
        }

        let (tmp, path, vectors) = alias_fixture();
        let parsed = vec![ParsedFile {
            path: "aliases.rs".into(),
            symbols: [1, 10, 20]
                .into_iter()
                .map(|line| ParsedSymbol {
                    name: if line == 20 { "other" } else { "same" }.into(),
                    kind: SymbolKind::Function,
                    line,
                    signature: Some("fn same()".into()),
                    doc: None,
                    body_tokens: Some("return".into()),
                })
                .collect(),
            refs: vec![],
            call_edges: vec![],
            bound_refs: vec![],
            skeletons: vec![],
            cpp_includes: vec![],
            trigram_bloom: None,
            hierarchy_captures: vec![],
        }];
        let context = |sym: &ParsedSymbol| {
            crate::embed::build_context(
                sym.kind.as_str(),
                &sym.name,
                &parsed[0].path,
                sym.signature.as_deref(),
                sym.doc.as_deref(),
                sym.body_tokens.as_deref(),
                256,
            )
        };
        assert_eq!(
            context(&parsed[0].symbols[0]),
            context(&parsed[0].symbols[1])
        );
        let index_path = tmp.path().join("index.vex");
        crate::store::writer::write_index_full(&parsed, &vectors, 3, &index_path).unwrap();
        let reader = IndexReader::open(&index_path).unwrap();

        let semantic = search_with_embedder(&reader, &mut Query, "same", 1, &path, true).unwrap();
        assert_eq!(semantic.iter().map(|r| r.line).collect::<Vec<_>>(), [1, 10]);
        let similar =
            crate::search::similar::find_similar(&reader, &path, "same", 5, 0.99, true).unwrap();
        assert_eq!(similar.iter().map(|r| r.line).collect::<Vec<_>>(), [10]);
        let duplicates =
            crate::search::similar::find_duplicates(&reader, &path, 0.99, 2, 5, true).unwrap();
        assert_eq!(duplicates.len(), 1);
        assert_eq!((duplicates[0].0.line, duplicates[0].1.line), (1, 10));
    }

    #[test]
    fn hnsw_rejects_invalid_or_incomplete_sidecars() {
        let (tmp, path, _) = alias_fixture();
        let sidecar = tmp.path().join("index.hashes");
        let valid_sidecar = std::fs::read(&sidecar).unwrap();
        for bytes in [
            b"GARBAGE".as_slice(),
            &valid_sidecar[..valid_sidecar.len() - 1],
        ] {
            std::fs::write(&sidecar, bytes).unwrap();
            assert!(HnswHandle::open(&path, 3, 3).is_none());
        }
        hash_index::save(&sidecar, &[11, 22]).unwrap();
        assert!(HnswHandle::open(&path, 3, 3).is_none());
        hash_index::save(&sidecar, &[11, 11, 11]).unwrap();
        assert!(HnswHandle::open(&path, 3, 3).is_none());
        std::fs::remove_file(&sidecar).unwrap();
        assert!(HnswHandle::open(&path, 3, 3).is_none());
        std::fs::write(&sidecar, valid_sidecar).unwrap();
        std::fs::write(&path, b"GARBAGE").unwrap();
        assert!(HnswHandle::open(&path, 3, 3).is_none());
    }

    #[test]
    fn identical_vectors_have_similarity_1() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn orthogonal_vectors_have_similarity_0() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn normalize_in_place_produces_unit_vector() {
        let mut v = vec![3.0, 4.0]; // norm = 5
        normalize_in_place(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_in_place_zero_vector_stays_zero() {
        // Guard against NaN — without the `if norm > 0.0` short-circuit
        // we'd divide by zero and produce a NaN-laden vector that
        // downstream `dot_product` would propagate as a similarity score.
        let mut v = vec![0.0_f32; 4];
        normalize_in_place(&mut v);
        assert!(v.iter().all(|x| *x == 0.0));
    }

    /// `dot_product` on two normalized vectors must equal
    /// `cosine_similarity` within float epsilon — this is the
    /// equivalence the v1.13 P5 fast path leans on. If a future refactor
    /// breaks the relationship (e.g. by changing one fn's reduction
    /// order), this test fails before the result reaches users.
    #[test]
    fn dot_product_equals_cosine_for_normalized() {
        let cases: &[(Vec<f32>, Vec<f32>)] = &[
            (vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]), // identical
            (vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]), // orthogonal
            (vec![1.0, 2.0, 3.0, 4.0], vec![4.0, 3.0, 2.0, 1.0]),
            (vec![0.1, -0.5, 0.7, 0.2], vec![-0.3, 0.6, 0.1, 0.8]),
        ];
        for (a_raw, b_raw) in cases {
            let mut a = a_raw.clone();
            let mut b = b_raw.clone();
            normalize_in_place(&mut a);
            normalize_in_place(&mut b);
            let cosine = cosine_similarity(&a, &b);
            let dot = dot_product(&a, &b);
            assert!(
                (cosine - dot).abs() < 1e-6,
                "dot_product diverged from cosine_similarity for normalized inputs: \
                 cosine={cosine}, dot={dot}, a={a:?}, b={b:?}"
            );
        }
    }

    #[test]
    fn hnsw_missing_returns_none() {
        let path = Path::new("/nonexistent/index.hnsw");
        let query = vec![1.0, 2.0, 3.0];
        assert!(search_hnsw_at(path, &query, 10, 100).is_none());
    }
}
