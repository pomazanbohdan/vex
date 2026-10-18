use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};

use crate::index::manifest::Manifest;
use crate::index::symbols::{ParsedFile, ParsedSymbol, SymbolKind};
use crate::store::reader::IndexReader;

pub(super) struct MigrationVector {
    symbol: ParsedSymbol,
    context: String,
    hash: u64,
    vector: Vec<f32>,
}

pub(super) struct MigratedFiles {
    pub(super) files: Vec<ParsedFile>,
    pub(super) vectors: Vec<Vec<f32>>,
}

pub(super) fn collect_migration_vectors(
    reader: &IndexReader,
    manifest: &Manifest,
    changed: &HashSet<&str>,
    body_tokens: Option<&[Option<String>]>,
    hash_path: &Path,
) -> Option<(String, HashMap<String, Vec<MigrationVector>>)> {
    let id = manifest.embedder_id.as_deref()?;
    let dim = crate::embed::embedder_dim(id)? as usize;
    if dim != reader.header().vector_dim as usize || manifest.vectors_normalized != Some(true) {
        return None;
    }
    let budget = crate::embed::embedder_char_budget(id)?;
    let bodies = body_tokens?;
    let hashes = crate::search::hash_index::load(hash_path).ok()?;
    if bodies.len() != reader.symbol_count() || hashes.len() != reader.symbol_count() {
        return None;
    }

    let mut files: HashMap<String, Vec<MigrationVector>> = HashMap::new();
    let mut invalid_files = HashSet::new();
    for (i, &hash) in hashes.iter().enumerate() {
        let record = reader.symbol(i)?;
        let path = reader.read_string(record.file_offset);
        if changed.contains(path) {
            continue;
        }
        let vector = reader.vector(record.vector_index)?.to_vec();
        if vector.len() != dim || vector.iter().any(|v| !v.is_finite()) {
            return None;
        }
        if invalid_files.contains(path) {
            continue;
        }
        let signature = reader.read_string(record.signature_offset);
        let symbol = ParsedSymbol {
            name: reader.read_string(record.name_offset).to_string(),
            kind: SymbolKind::try_from(record.kind).ok()?,
            line: record.line as usize,
            signature: (!signature.is_empty()).then(|| signature.to_string()),
            // SymbolRecord does not persist docs; a context mismatch declines this file.
            doc: None,
            body_tokens: bodies[i].clone(),
        };
        let context = crate::embed::build_context(
            symbol.kind.as_str(),
            &symbol.name,
            path,
            symbol.signature.as_deref(),
            None,
            symbol.body_tokens.as_deref(),
            budget,
        );
        if crate::embed::cache::context_hash(id, &context) != hash {
            invalid_files.insert(path.to_string());
            files.remove(path);
            continue;
        }
        files
            .entry(path.to_string())
            .or_default()
            .push(MigrationVector {
                symbol,
                context,
                hash,
                vector,
            });
    }
    Some((id.to_string(), files))
}

pub(super) fn reuse_migration_vectors(
    fresh_files: Vec<ParsedFile>,
    embedder_id: &str,
    mut old_files: HashMap<String, Vec<MigrationVector>>,
) -> Result<MigratedFiles> {
    let budget = crate::embed::embedder_char_budget(embedder_id)
        .context("migration embedder has no known context budget")?;
    let mut retained_files = Vec::new();
    let mut retained_vectors = Vec::new();
    let mut unembedded_files = Vec::new();

    for fresh in fresh_files {
        let Some(old) = old_files.remove(&fresh.path) else {
            unembedded_files.push(fresh);
            continue;
        };
        let mut old_by_key = HashMap::new();
        let mut ambiguous = HashSet::new();
        for (idx, prior) in old.iter().enumerate() {
            let key = (prior.symbol.name.as_str(), prior.symbol.kind);
            if old_by_key.insert(key, idx).is_some() {
                ambiguous.insert(key);
            }
        }
        let mut matched = Vec::with_capacity(fresh.symbols.len());
        let mut used = HashSet::new();
        let mut valid = old.len() == fresh.symbols.len();
        for symbol in &fresh.symbols {
            let key = (symbol.name.as_str(), symbol.kind);
            let Some(&idx) = old_by_key.get(&key) else {
                valid = false;
                break;
            };
            if ambiguous.contains(&key) || !used.insert(idx) {
                valid = false;
                break;
            }
            let prior = &old[idx];
            let context = crate::embed::build_context(
                symbol.kind.as_str(),
                &symbol.name,
                &fresh.path,
                symbol.signature.as_deref(),
                symbol.doc.as_deref(),
                symbol.body_tokens.as_deref(),
                budget,
            );
            if context != prior.context
                || crate::embed::cache::context_hash(embedder_id, &context) != prior.hash
            {
                valid = false;
                break;
            }
            matched.push(idx);
        }
        if valid && matched.len() == old.len() {
            let mut vectors = old
                .into_iter()
                .map(|prior| Some(prior.vector))
                .collect::<Vec<_>>();
            for idx in matched {
                retained_vectors.push(
                    vectors
                        .get_mut(idx)
                        .and_then(Option::take)
                        .context("migration match has no stored vector")?,
                );
            }
            retained_files.push(fresh);
        } else {
            unembedded_files.push(fresh);
        }
    }
    retained_files.extend(unembedded_files);
    Ok(MigratedFiles {
        files: retained_files,
        vectors: retained_vectors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symbol(name: &str, body: &str) -> ParsedSymbol {
        ParsedSymbol {
            name: name.to_string(),
            kind: SymbolKind::Function,
            line: 1,
            signature: None,
            doc: None,
            body_tokens: Some(body.to_string()),
        }
    }

    fn file(path: &str, symbols: Vec<ParsedSymbol>) -> ParsedFile {
        ParsedFile {
            path: path.to_string(),
            symbols,
            refs: Vec::new(),
            call_edges: Vec::new(),
            bound_refs: Vec::new(),
            skeletons: Vec::new(),
            cpp_includes: Vec::new(),
            trigram_bloom: None,
            hierarchy_captures: Vec::new(),
        }
    }

    fn prior(path: &str, symbol: ParsedSymbol, vector: Vec<f32>) -> MigrationVector {
        let embedder = "minilm-l6-v2";
        let context = crate::embed::build_context(
            symbol.kind.as_str(),
            &symbol.name,
            path,
            symbol.signature.as_deref(),
            symbol.doc.as_deref(),
            symbol.body_tokens.as_deref(),
            crate::embed::embedder_char_budget(embedder).unwrap(),
        );
        MigrationVector {
            hash: crate::embed::cache::context_hash(embedder, &context),
            symbol,
            context,
            vector,
        }
    }

    #[test]
    fn context_count_and_ambiguous_identity_decline_only_their_file() {
        let embedder = "minilm-l6-v2";
        let fresh_files = vec![
            file("context.rs", vec![symbol("changed", "new body")]),
            file(
                "count.rs",
                vec![symbol("first", "one"), symbol("second", "two")],
            ),
            file(
                "ambiguous.rs",
                vec![symbol("duplicate", "same"), symbol("duplicate", "same")],
            ),
            file("eligible.rs", vec![symbol("retained", "stable")]),
        ];
        let old_files = HashMap::from([
            (
                "context.rs".to_string(),
                vec![prior(
                    "context.rs",
                    symbol("changed", "old body"),
                    vec![0.0, 1.0],
                )],
            ),
            (
                "count.rs".to_string(),
                vec![prior("count.rs", symbol("first", "one"), vec![0.0, 1.0])],
            ),
            (
                "ambiguous.rs".to_string(),
                vec![
                    prior("ambiguous.rs", symbol("duplicate", "same"), vec![0.0, 1.0]),
                    prior("ambiguous.rs", symbol("duplicate", "same"), vec![1.0, 0.0]),
                ],
            ),
            (
                "eligible.rs".to_string(),
                vec![prior(
                    "eligible.rs",
                    symbol("retained", "stable"),
                    vec![1.0, -0.0],
                )],
            ),
        ]);

        let migrated = reuse_migration_vectors(fresh_files, embedder, old_files).unwrap();

        assert_eq!(
            migrated
                .files
                .iter()
                .map(|file| file.path.as_str())
                .collect::<Vec<_>>(),
            ["eligible.rs", "context.rs", "count.rs", "ambiguous.rs"]
        );
        assert_eq!(migrated.vectors.len(), 1);
        assert_eq!(
            migrated.vectors[0]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [1.0f32.to_bits(), (-0.0f32).to_bits()]
        );
    }
}
