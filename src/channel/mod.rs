//! Reference channels — the abstraction behind `vex impact`'s
//! delete-safety report and (Phase 2) `vex usages`'s binder-vs-FST
//! split. Each channel resolves the question "what references this
//! symbol?" via a different mechanism (scope-binder, FST identifier
//! scan, grep word-boundary regex, call-graph indegree) and reports
//! its findings via a shared [`ChannelResult`] envelope.
//!
//! v1.21.0 architecture (extracted from the v1.20.0 inline 4-channel
//! body of `cmd_impact.rs`). Verdict logic is now data-driven over
//! `&[ChannelInvocation]` so adding a fifth channel does not require
//! touching [`derive_verdict`] — only appending another entry to the
//! channel slice in `cmd_impact::DEFAULT_CHANNELS`.
//!
//! ## Tier classification
//!
//! Channels split into two tiers (see [`ChannelTier`]):
//!
//! - **`Binder`** — evidence is binder/graph-confirmed. A hit from a
//!   binder channel implies a real reference; the verdict goes
//!   `Unsafe` (Phase 11.1 scope binder, Phase 10.2 call graph).
//! - **`Text`** — evidence is text-only (FST identifier match or
//!   regex scan). A hit might be a comment, string literal, or
//!   dynamic-dispatch reference that binders can't see; verdict is
//!   `Uncertain` when only text channels hit.
//!
//! ## Adding a new channel
//!
//! 1. Define a zero-sized type that implements [`Channel`].
//! 2. Pick the right [`ChannelTier`] — `Binder` if the channel can
//!    confirm "this is definitely a real reference" (e.g. when a new
//!    binder lands for Go), `Text` otherwise.
//! 3. Append the channel to [`DEFAULT_CHANNELS`] in
//!    `src/cli/cmd_impact.rs`.
//!
//! That's it. [`derive_verdict`] is data-driven — no edits needed
//! when channel count goes from 4 to 5 to N. The wire format
//! (`results.channels.<name>: ChannelResult`) is still struct-keyed
//! on the receiver side (`ImpactChannels`); see
//! `cmd_impact::ImpactChannels::from_invocations` for the bridge.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::cli::scope::PathScope;
use crate::store::reader::IndexReader;
use crate::util::paths::to_rel_posix;

mod dart;
mod dart_server;

/// Cap on rows surfaced in each channel's `sample`. Verdict only
/// needs `≥1 vs 0` granularity; the sample exists so the agent can
/// see *where* the hits are without re-running a per-channel query.
pub const SAMPLE_LIMIT: usize = 10;

/// Hard cap on the grep channel — `\b<Name>\b` against the whole
/// project should never produce thousands of hits for a real symbol
/// name, but the regex scans every file so a bound matters.
pub const GREP_HARD_CAP: usize = 500;

/// Tier classification used by [`derive_verdict`] to know whether
/// a non-zero hit confirms (`Binder`) or only hints (`Text`) at real
/// usage. Binder-tier channels resolve references through the
/// scope-binder or call-graph index, so a non-zero count means
/// "definitely referenced". Text-tier channels see identifier
/// matches anywhere — comments, string literals, configuration
/// files — so a non-zero count means "maybe referenced".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelTier {
    Binder,
    Text,
}

/// A single (path, line) location surfaced by a channel.
#[derive(Debug, Serialize)]
pub struct HitLocation {
    pub path: String,
    pub line: u32,
}

/// Per-category drop counters reported by each channel. Used by
/// `vex usages --why` to attribute filter losses to the right cause
/// (def-site vs prose path vs scope glob) so the agent can see why
/// `hits_before_filter` shrank to `hits_after_filter`.
///
/// `scope` counts rows dropped by `path_scope` (include/exclude
/// globs); `def_site` counts the symbol's own declaration row
/// (controlled by `ChannelContext::filter_def_sites`); `docs` counts
/// prose-path rows (controlled by `ChannelContext::exclude_docs`).
/// Drops accumulate in that order — a row that fails `scope` does
/// not also bump `def_site` even if it would have.
#[derive(Debug, Default, Clone, Copy)]
pub struct DropCounts {
    pub def_site: usize,
    pub docs: usize,
    pub scope: usize,
}

/// Primary channel output: every surviving hit (un-truncated, in
/// channel order) plus per-category drop counters and the raw
/// pre-filter count. `ChannelResult` wraps this for the impact wire
/// envelope; cmd_usages reads it directly to paginate, apply
/// command-specific filters (filter_path, diff), and emit the
/// `--why` trace.
#[derive(Debug)]
pub struct ChannelOutput {
    /// Semantic backend provenance; absent for persisted native binder edges.
    pub backend: Option<serde_json::Value>,
    pub available: bool,
    pub unavailable_reason: Option<String>,
    /// Raw fetch count for THIS channel's domain, before any
    /// `path_scope` / `def_site` / `docs` filter ran. Survivor count
    /// = `hits.len()`; total drops attributable to those three
    /// filters = `pre_filter_count - hits.len()` minus any
    /// channel-specific exclusions (see below). Note: each channel
    /// defines its own "domain" — e.g. `TransitiveCallersChannel`
    /// excludes depth-1 rows from `pre_filter_count` because those
    /// callers belong to `CallGraphCallersChannel`, not here. Trace
    /// consumers should NOT cross-sum `pre_filter_count` across
    /// channels expecting a single global "raw fetch" view.
    pub pre_filter_count: usize,
    /// Post-filter hits in channel-native order. NOT capped to
    /// `SAMPLE_LIMIT` — callers that want the impact wire envelope
    /// take the cap via `ChannelResult::from_output`.
    pub hits: Vec<HitLocation>,
    pub dropped: DropCounts,
}

impl ChannelOutput {
    pub fn unavailable(reason: &str) -> Self {
        Self {
            backend: None,
            available: false,
            unavailable_reason: Some(reason.to_string()),
            pre_filter_count: 0,
            hits: Vec::new(),
            dropped: DropCounts::default(),
        }
    }
}

/// Output of a channel run: whether it could execute, how many hits,
/// a bounded sample, and a truncation marker. Serialized to the wire
/// as `ImpactChannels.<name>: ChannelResult` so the field shape is
/// part of the v1 envelope contract.
#[derive(Debug, Serialize)]
pub struct ChannelResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<serde_json::Value>,
    /// `false` when the channel could not run (e.g. the index lacks
    /// the requisite section). Verdict logic treats `available:
    /// false` as informationless — an unavailable binder channel
    /// reporting `count: 0` does NOT drag the verdict to `Safe`.
    pub available: bool,
    /// Human-readable reason when `available == false`. Skipped from
    /// JSON when `None` to keep the envelope tidy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unavailable_reason: Option<String>,
    /// Total post-filter hit count (sample may be smaller — see
    /// `truncated`).
    pub count: usize,
    /// First [`SAMPLE_LIMIT`] hits, surfaced so an agent can drill
    /// into specific files without re-running the per-channel query.
    pub sample: Vec<HitLocation>,
    /// `true` when `count > sample.len()` — the rest is omitted.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

impl ChannelResult {
    /// Test-only convenience: build an unavailable `ChannelResult`
    /// (mirrors the old pre-Phase-2 shape). Production paths go
    /// through [`Self::from_output`] which lifts a
    /// [`ChannelOutput::unavailable`].
    #[cfg(test)]
    pub fn unavailable(reason: &str) -> Self {
        Self {
            backend: None,
            available: false,
            unavailable_reason: Some(reason.to_string()),
            count: 0,
            sample: Vec::new(),
            truncated: false,
        }
    }

    /// Test-only convenience: build a successful `ChannelResult`
    /// from a raw hit list (mirrors the old pre-Phase-2 shape).
    /// Production paths go through [`Self::from_output`] which
    /// preserves `ChannelOutput`'s drop counters for `vex usages`.
    #[cfg(test)]
    pub fn from_hits(mut hits: Vec<HitLocation>) -> Self {
        let count = hits.len();
        let truncated = count > SAMPLE_LIMIT;
        hits.truncate(SAMPLE_LIMIT);
        Self {
            backend: None,
            available: true,
            unavailable_reason: None,
            count,
            sample: hits,
            truncated,
        }
    }

    /// Lift a [`ChannelOutput`] into the impact wire-envelope shape:
    /// take the first [`SAMPLE_LIMIT`] hits and mark `truncated` when
    /// more were dropped. Unavailable outputs pass through unchanged.
    pub fn from_output(out: ChannelOutput) -> Self {
        if !out.available {
            return Self {
                backend: out.backend,
                available: false,
                unavailable_reason: out.unavailable_reason,
                count: 0,
                sample: Vec::new(),
                truncated: false,
            };
        }
        let count = out.hits.len();
        let truncated = count > SAMPLE_LIMIT;
        let mut sample = out.hits;
        sample.truncate(SAMPLE_LIMIT);
        Self {
            backend: out.backend,
            available: true,
            unavailable_reason: None,
            count,
            sample,
            truncated,
        }
    }
}

/// Borrowed inputs every channel needs to run. Built once by the
/// caller and shared across all channels for a single `vex impact`
/// invocation.
pub struct ChannelContext<'a> {
    pub reader: &'a IndexReader,
    pub root: &'a Path,
    pub symbol: &'a str,
    pub file_paths: &'a [String],
    /// Def-site map used by FST + grep channels to strip the
    /// symbol's own declaration row (declarations are not "uses").
    /// Pre-computed once via [`build_def_sites`] so each channel
    /// avoids duplicate symbol-FST lookups. Borrow-friendly shape
    /// (Phase 2 H2): the channel checks `def_sites.get(path).map_or(false,
    /// |lines| lines.contains(&line))` — no `String` clone per row.
    pub def_sites: &'a HashMap<&'a str, HashSet<u32>>,
    pub path_scope: &'a PathScope,
    pub excludes: &'a [String],
    /// Whether channels should strip the symbol's own definition row.
    /// `vex impact` uses `true` (a declaration is not a "use"); the
    /// `--include-self` escape hatch on `vex usages` toggles it to
    /// `false`. Strict / call-graph channels ignore the flag (they
    /// exclude def-sites by construction).
    pub filter_def_sites: bool,
    /// Opt-in: when true, FST + grep channels drop hits in
    /// `*.md`/`*.markdown`/`*.txt`/`*.rst`/`*.adoc` paths
    /// (see [`crate::util::paths::is_doc_path`]). v1.20.1 (D4
    /// parity with `vex search --code-only`). Off by default —
    /// the `Uncertain`-verdict story for prose-only mentions
    /// (e.g. a name appearing only in CHANGELOG) is the whole
    /// point of `vex impact`; this flag opts out for agents that
    /// want a code-only blast radius. cmd_usages mirrors this
    /// (`--include-docs` to opt back in).
    pub exclude_docs: bool,
    /// BFS hop budget for transitive-callers traversal. v1.21.0:
    /// `vex impact --depth N` lets agents see the full upstream
    /// blast radius up to N call-graph hops. `1` (default) is
    /// "direct callers only" — `TransitiveCallersChannel` reports
    /// `unavailable` since its job is depth ≥ 2. Bounded at 16
    /// elsewhere to keep BFS bounded on monorepos.
    pub depth: u32,
}

/// Trait implemented by each reference channel. Returns
/// `Ok(ChannelOutput)` even when the channel could not run — that
/// shape carries the full hit list, the raw pre-filter count, and
/// per-category drop counters. Callers (cmd_impact, cmd_usages)
/// choose how to present it: `vex impact` lifts via
/// [`ChannelResult::from_output`] for the wire envelope (`count` +
/// `sample[:SAMPLE_LIMIT]`); `vex usages` reads `hits` directly to
/// paginate and emit the `--why` trace. Reserve `Err` for genuinely
/// unexpected failures (I/O errors, regex compile failures) that
/// should abort the whole query.
pub trait Channel: Sync {
    fn name(&self) -> &'static str;
    fn tier(&self) -> ChannelTier;
    fn run(&self, ctx: &ChannelContext<'_>) -> Result<ChannelOutput>;
}

/// Data-driven verdict input: one slot per channel that ran. The
/// caller builds this vector by mapping `Channel::run` across its
/// channel list; [`derive_verdict`] never reads the channel impls
/// themselves, only this DTO. That's what lets a 5th channel land
/// without touching the verdict code.
pub struct ChannelInvocation {
    pub name: &'static str,
    pub tier: ChannelTier,
    pub result: ChannelResult,
}

/// Build the `path → {line_numbers}` map of the queried symbol's own
/// definitions. Used by FST + grep channels to strip declarations
/// from their hit counts (a declaration is not a "use"). Shared so
/// the symbol-FST lookup happens once per query, not per channel.
///
/// **H2 (Phase 2)**: the map is keyed on `&str` borrowed from
/// `IndexReader::read_string`, so each channel does
/// `def_sites.get(path_str).is_some_and(|lines| lines.contains(&line))`
/// without cloning the path on every row. Pre-fix the channels paid
/// two `String` clones per fetched row (one for the def-site lookup
/// tuple, one for the surviving `HitLocation`); now only the
/// surviving rows pay one clone.
///
/// Path normalisation invariant: the strings here come from
/// `read_string(sym.file_offset)`, which the writer routed through
/// `util::paths::to_rel_posix`. Match against `file_paths.get(...)`
/// values is byte-identical on every platform.
#[must_use]
pub fn build_def_sites<'a>(
    reader: &'a IndexReader,
    symbol: &str,
) -> HashMap<&'a str, HashSet<u32>> {
    let mut map: HashMap<&'a str, HashSet<u32>> = HashMap::new();
    let full_dart_name = symbol.to_lowercase();
    if let Some(sym_fst) = reader.symbol_fst_reader() {
        for sym_idx in sym_fst.find(symbol) {
            if let Some(sym) = reader.symbol(sym_idx as usize) {
                let file_path: &str = reader.read_string(sym.file_offset);
                if file_path.ends_with(".dart")
                    && reader.read_string(sym.name_offset).to_lowercase() != full_dart_name
                {
                    continue;
                }
                map.entry(file_path).or_default().insert(sym.line);
            }
        }
    }
    map
}

/// `true` when the (path, line) coordinate is in the def-site set.
/// Helper so each channel doesn't repeat the
/// `Option<&HashSet>::map_or` dance.
#[inline]
fn is_def_site(def_sites: &HashMap<&str, HashSet<u32>>, path: &str, line: u32) -> bool {
    def_sites
        .get(path)
        .is_some_and(|lines| lines.contains(&line))
}

/// Three-valued impact verdict. Wire serialisation uses snake_case
/// (`"safe"`, `"unsafe"`, `"uncertain"`) to match the
/// `results.verdict` field documented in `docs/MCP-SCHEMA.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Safe,
    Unsafe,
    Uncertain,
}

/// Decide the verdict from the channel invocation list. Data-driven
/// over [`ChannelInvocation`] — adding a fifth channel does not
/// require touching this function, only appending to the channel
/// slice the caller passes to `Channel::run`.
///
/// Rules (see `docs/LIMITATIONS.md` §6 for the user-facing version):
///
/// - **`Unsafe`** ⇔ any `Binder`-tier channel reported `available:
///   true && count > 0`. Binder/graph confirmed real usage.
/// - **`Uncertain`** ⇔ only `Text`-tier channels hit (likely
///   string-literal / comment / dynamic-dispatch mentions) **OR**
///   every binder channel reported `available: false` (no binder
///   evidence either way — the verdict is informationless without
///   re-indexing).
/// - **`Safe`** ⇔ every channel that ran reported zero hits AND at
///   least one binder channel was available to confirm. An
///   unavailable binder channel's `count: 0` is not a confirmation.
///
/// `verdict_explanation` enumerates which channels actually ran and
/// which were unavailable so the agent can read the conclusion
/// without a false "all channels reported zero" claim when binder
/// sections are missing from a pre-v1.8 index.
#[must_use]
pub fn derive_verdict(invocations: &[ChannelInvocation]) -> (Verdict, String) {
    let mut binder_confirms: Vec<(&str, usize)> = Vec::new();
    let mut text_hits: Vec<(&str, usize)> = Vec::new();
    let mut unavailable: Vec<&str> = Vec::new();
    let mut available_names: Vec<&str> = Vec::new();
    let mut any_binder_available = false;

    for inv in invocations {
        if !inv.result.available {
            unavailable.push(inv.name);
            continue;
        }
        available_names.push(inv.name);
        match inv.tier {
            ChannelTier::Binder => {
                any_binder_available = true;
                if inv.result.count > 0 {
                    binder_confirms.push((inv.name, inv.result.count));
                }
            }
            ChannelTier::Text => {
                if inv.result.count > 0 {
                    text_hits.push((inv.name, inv.result.count));
                }
            }
        }
    }

    let unavailable_note = if unavailable.is_empty() {
        String::new()
    } else {
        format!(
            " (unavailable: {} — re-run `vex index` for stronger evidence)",
            unavailable.join(", ")
        )
    };

    if !binder_confirms.is_empty() {
        let reasons = binder_confirms
            .iter()
            .map(|(n, c)| format!("{n}={c}"))
            .collect::<Vec<_>>()
            .join(", ");
        return (
            Verdict::Unsafe,
            format!(
                "binder/graph confirmed real usage ({reasons}). Do not delete without rewriting call sites.",
            ),
        );
    }

    if !text_hits.is_empty() {
        let reasons = text_hits
            .iter()
            .map(|(n, c)| format!("{n}={c}"))
            .collect::<Vec<_>>()
            .join(", ");
        return (
            Verdict::Uncertain,
            format!(
                "text-only matches surfaced ({reasons}) but binder/call-graph saw none{unavailable_note}. \
                 Likely string-literal mentions, comments, or dynamic dispatch — manual inspection required.",
            ),
        );
    }

    if !any_binder_available {
        return (
            Verdict::Uncertain,
            "text channels reported zero hits, but no binder channel ran on this index \
             (pre-v1.8 / pre-Phase 10.2, or no binder coverage for this project's languages). \
             Re-run `vex index` to rebuild before relying on a verdict."
                .to_string(),
        );
    }

    (
        Verdict::Safe,
        format!(
            "{} of {} channels reported zero hits ({}){unavailable_note}. \
             Delete is highly likely safe.",
            available_names.len(),
            invocations.len(),
            available_names.join(", "),
        ),
    )
}

// ───────────────────────── Channel implementations ────────────────────────

/// Native v5 `reference_edges` plus live Dart Analysis Server references.
/// Native edges cover Rust / TypeScript / Python / C# / C++ / Go / Java / Kotlin;
/// Dart requires an available SDK and fresh project source. Unsupported domains
/// report unavailable rather than substituting textual references.
pub struct StrictRefsChannel;

impl Channel for StrictRefsChannel {
    fn name(&self) -> &'static str {
        "strict_refs"
    }
    fn tier(&self) -> ChannelTier {
        ChannelTier::Binder
    }
    fn run(&self, ctx: &ChannelContext<'_>) -> Result<ChannelOutput> {
        let dart_result = if dart::has_dart(ctx) {
            match dart::run(ctx) {
                Ok((hits, backend)) => Some(dart::output(hits, ctx, backend)),
                Err(err) => {
                    return Ok(ChannelOutput::unavailable(&format!(
                        "Dart strict backend unavailable: {err:#}"
                    )))
                }
            }
        } else {
            None
        };
        if !ctx.reader.has_ref_edges() {
            if let Some(result) = dart_result {
                return Ok(result);
            }
            return Ok(ChannelOutput::unavailable(
                "index has no v5 reference_edges section (pre-v1.8 index, or no binder coverage \
                 for this project's languages) — re-run `vex index` to rebuild",
            ));
        }
        let sym_fst = ctx
            .reader
            .symbol_fst_reader()
            .context("symbol FST missing — re-run `vex index` to rebuild")?;
        // Binder edges exclude def-sites and doc paths by construction
        // (scope-binder walks AST nodes inside source files), so
        // `filter_def_sites` and `exclude_docs` are no-ops here. Only
        // `path_scope` applies, which we count as `dropped.scope`.
        let mut hits = Vec::new();
        let mut pre_filter_count = 0usize;
        let mut dropped = DropCounts::default();
