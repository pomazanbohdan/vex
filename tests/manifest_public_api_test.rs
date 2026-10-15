use std::collections::HashMap;

use vex::index::incremental_state::IncrementalState;
use vex::index::manifest::Manifest;

#[test]
fn manifest_existing_literal_still_compiles() {
    let manifest = Manifest {
        files: HashMap::new(),
        git_head: None,
        indexed_at: None,
        embedder_id: None,
        call_graph: None,
        bm25: None,
        pattern_index: None,
        pattern_index_full: None,
        clusters_full: None,
        clusters_opt_out: None,
        vectors_normalized: None,
        trigram_persisted: None,
        rename_chains_built: None,
        rename_chains_minilm_tiebreak_hits: None,
        state: IncrementalState::default(),
    };
    assert!(manifest.files.is_empty());
}
