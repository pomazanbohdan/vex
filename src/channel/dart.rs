use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use serde_json::{json, Value};

use super::dart_server::{discover_sdk, normal_path, Server};
use super::{ChannelContext, ChannelOutput, DropCounts, HitLocation};
use crate::index::hasher::hash_file;
use crate::index::symbols::SymbolKind;
use crate::parse::language::Language;
use crate::util::paths::to_rel_posix;

#[derive(Debug)]
struct Target {
    file: PathBuf,
    name: String,
    line: u32,
    kind: SymbolKind,
}

struct FileReferences {
    path: String,
    locations: Vec<(u64, u32)>,
}

fn exact_definitions(ctx: &ChannelContext<'_>) -> Option<Vec<u32>> {
    let query = ctx.symbol.to_lowercase();
    Some(
        ctx.reader
            .symbol_fst_reader()?
            .find(ctx.symbol)
            .into_iter()
            .filter(|&id| {
                ctx.reader.symbol(id as usize).is_some_and(|symbol| {
                    ctx.reader.read_string(symbol.name_offset).to_lowercase() == query
                })
            })
            .collect(),
    )
}

pub(super) fn has_dart(ctx: &ChannelContext<'_>) -> bool {
    if !ctx.file_paths.iter().any(|p| p.ends_with(".dart")) {
        return false;
    }
    let Some(definitions) = exact_definitions(ctx) else {
        return true;
    };
    definitions.is_empty()
        || definitions.into_iter().any(|id| {
            ctx.reader.symbol(id as usize).is_some_and(|symbol| {
                ctx.reader
                    .read_string(symbol.file_offset)
                    .ends_with(".dart")
            })
        })
}

pub(super) fn run(ctx: &ChannelContext<'_>) -> Result<(Vec<HitLocation>, Option<Value>)> {
    let start = Instant::now();
    let root = normal_path(ctx.root)?;
    let mut targets = Vec::new();
    for id in exact_definitions(ctx).context("symbol FST missing")? {
        let symbol = ctx
            .reader
            .symbol(id as usize)
            .context("indexed symbol missing")?;
        let path = ctx.reader.read_string(symbol.file_offset);
        if !path.ends_with(".dart") {
            continue;
        }
        targets.push(Target {
            file: normal_path(&root.join(path))?,
            name: ctx.reader.read_string(symbol.name_offset).to_owned(),
            line: symbol.line,
            kind: SymbolKind::try_from(symbol.kind)
                .map_err(|kind| anyhow::anyhow!("invalid symbol kind {kind}"))?,
        });
    }
    let snapshot = Snapshot::capture(ctx, &root)?;
    if targets.is_empty() {
        snapshot.verify(ctx, &root, &|| Ok(()))?;
        return Ok((Vec::new(), None));
    }
    let dart = discover_sdk(&root)?;
    let mut files: Vec<_> = targets.iter().map(|t| t.file.clone()).collect();
    files.sort();
    files.dedup();
    let cache_base = crate::util::config::platform_cache_base();
    anyhow::ensure!(cache_base.is_absolute(), "user cache directory unavailable");
    let cache = cache_base.join("vex/dart-analysis");
    std::fs::create_dir_all(&cache)?;
    let mut server = Server::start(&dart, &cache)?;
    let excluded: Vec<_> = [".fvm", ".git", "node_modules", ".vex_cache"]
        .into_iter()
        .map(|p| root.join(p))
        .filter(|p| p.is_dir())
        .collect();
    server.analyze(&root, &files, &excluded)?;
    let indexed_paths: std::collections::HashSet<_> =
        ctx.file_paths.iter().map(String::as_str).collect();
    let mut identities = BTreeSet::new();
    let mut requests = Vec::new();
    let mut declarations = Vec::new();
    let mut targets_by_file: BTreeMap<PathBuf, Vec<Target>> = BTreeMap::new();
    for target in targets {
        targets_by_file
            .entry(target.file.clone())
            .or_default()
            .push(target);
    }
    for (file, targets) in targets_by_file {
        server.check_deadline()?;
        let mut elements = Vec::new();
        collect_elements(server.outline(&file)?, &mut elements);
        for target in targets {
            let same_line_declarations: Vec<_> = elements
                .iter()
                .copied()
                .filter(|element| matches_target(element, &target))
                .cloned()
                .collect();
            anyhow::ensure!(
                !same_line_declarations.is_empty(),
                "Dart declaration {}:{} ({}) cannot map to SDK outline; refresh index",
                target.file.display(),
                target.line,
                target.name
            );
            for declaration in same_line_declarations {
                server.check_deadline()?;
                let offset = declaration["location"]["offset"]
                    .as_u64()
                    .context("declaration offset missing")?;
                let identity = (
                    target.file.clone(),
                    offset,
                    declaration["kind"]
                        .as_str()
                        .context("declaration kind missing")?
                        .to_owned(),
                );
                if !identities.insert(identity) {
                    continue;
                }
                requests.push((target.file.clone(), offset));
                declarations.push(declaration);
            }
        }
    }
    let mut references_by_file: BTreeMap<PathBuf, FileReferences> = BTreeMap::new();
    for (request_batch, declaration_batch) in requests.chunks(16).zip(declarations.chunks(16)) {
        server.check_deadline()?;
        let responses = server.references_many(request_batch)?;
        anyhow::ensure!(
            responses.len() == request_batch.len(),
            "Dart reference response count mismatch"
        );
        for (((file, _), declaration), (element, references)) in
            request_batch.iter().zip(declaration_batch).zip(responses)
        {
            server.check_deadline()?;
            ensure_resolved_declaration(&element, declaration, file)?;
            for reference in references {
                server.check_deadline()?;
                if reference["isPotential"] != false || reference["kind"] == "DECLARATION" {
                    continue;
                }
                let location = &reference["location"];
                let file = normal_path(Path::new(
                    location["file"]
                        .as_str()
                        .context("reference file missing")?,
                ))?;
                let Some(path) = to_rel_posix(&file, &root) else {
                    continue;
                };
                if !indexed_paths.contains(path.as_str()) {
                    continue;
                }
                let offset = location["offset"]
                    .as_u64()
                    .context("reference offset missing")?;
                let line = u32::try_from(
                    location["startLine"]
                        .as_u64()
                        .context("reference line missing")?,
                )?;
                references_by_file
                    .entry(file)
                    .or_insert_with(|| FileReferences {
                        path,
                        locations: Vec::new(),
                    })
                    .locations
                    .push((offset, line));
            }
        }
    }
    let mut hits = BTreeSet::new();
    for (
        file,
        FileReferences {
            path,
            mut locations,
        },
    ) in references_by_file
    {
        server.check_deadline()?;
        let source = std::fs::read_to_string(&file)?;
        for line in reference_lines(&source, &mut locations, &|| server.check_deadline())? {
            server.check_deadline()?;
            hits.insert((path.clone(), line));
        }
    }
    snapshot.verify(ctx, &root, &|| server.check_deadline())?;
    server.check_deadline()?;
    let backend = json!({"kind": "dart_analysis_server", "protocol_version": server.version, "sdk_executable": dart,
        "execution_runtime": server.runtime,
        "freshness": "live_source_hashes_verified_before_and_after", "elapsed_ms": start.elapsed().as_millis(),
        "potential_references": false, "scope": "indexed_project_files", "coverage": "compiler_resolved_only"});
    Ok((
        hits.into_iter()
            .map(|(path, line)| HitLocation { path, line })
            .collect(),
        Some(backend),
    ))
}

fn collect_elements<'a>(outline: &'a Value, elements: &mut Vec<&'a Value>) {
    if let Some(element) = outline.get("element") {
        elements.push(element);
    }
    if let Some(children) = outline["children"].as_array() {
        for child in children {
            collect_elements(child, elements);
        }
    }
}

fn matches_target(element: &Value, target: &Target) -> bool {
    if element["location"]["startLine"] != target.line {
        return false;
    }
    let name = element["name"].as_str().unwrap_or("");
    let kind = element["kind"].as_str().unwrap_or("");
    let same_name = name == target.name
        || (kind == "CONSTRUCTOR" && name.rsplit('.').next() == Some(target.name.as_str()));
    same_name
        && match target.kind {
            SymbolKind::Class | SymbolKind::Struct | SymbolKind::Interface | SymbolKind::Trait => {
                matches!(
                    kind,
                    "CLASS" | "MIXIN" | "EXTENSION" | "EXTENSION_TYPE" | "CLASS_TYPE_ALIAS"
                )
            }
            SymbolKind::Enum => kind == "ENUM",
            SymbolKind::Function => matches!(kind, "FUNCTION" | "GETTER" | "SETTER"),
            SymbolKind::Method => matches!(kind, "METHOD" | "CONSTRUCTOR" | "GETTER" | "SETTER"),
            SymbolKind::Property | SymbolKind::Constant => matches!(
                kind,
                "FIELD" | "TOP_LEVEL_VARIABLE" | "ENUM_CONSTANT" | "GETTER" | "SETTER"
            ),
            SymbolKind::TypeAlias => matches!(
                kind,
                "FUNCTION_TYPE_ALIAS" | "TYPE_ALIAS" | "CLASS_TYPE_ALIAS"
            ),
            _ => false,
        }
}

fn ensure_resolved_declaration(element: &Value, declaration: &Value, file: &Path) -> Result<()> {
    let same_file = normal_path(Path::new(
        element["location"]["file"]
            .as_str()
            .context("element file missing")?,
    ))? == file;
    let exact = element["location"]["offset"] == declaration["location"]["offset"]
        && element["kind"] == declaration["kind"];
    // SDK search normalizes accessors to their synthetic variable. Its nameOffset
    // is absent: the protocol converter reports offset 0 and the UTF-16 name length.
    let property = matches!(declaration["kind"].as_str(), Some("GETTER" | "SETTER"))
        && matches!(
            element["kind"].as_str(),
            Some("FIELD" | "TOP_LEVEL_VARIABLE")
        )
        && declaration["name"].as_str().is_some_and(|name| {
            element["name"] == name
                && element["location"]["length"].as_u64()
                    == Some(name.encode_utf16().count() as u64)
        })
        && element["location"]["offset"] == 0
        && element["location"]["startLine"] == 1
        && element["location"]["startColumn"] == 1;
    anyhow::ensure!(
        same_file && (exact || property),
        "Analysis Server resolved a different declaration"
    );
    Ok(())
}

fn reference_lines(
    source: &str,
    locations: &mut [(u64, u32)],
    check_deadline: &impl Fn() -> Result<()>,
) -> Result<Vec<u32>> {
    locations.sort_unstable();
    check_deadline()?;
    let mut chars = source.char_indices();
    let mut units = 0;
    let mut line = 1u32;
    let mut scanned = 0usize;
    let mut lines = Vec::with_capacity(locations.len());
    for &(offset, expected_line) in locations.iter() {
        check_deadline()?;
        while units < offset {
            let (_, ch) = chars
                .next()
                .context("Dart UTF-16 offset exceeds source length")?;
            units += ch.len_utf16() as u64;
            anyhow::ensure!(
                units <= offset,
                "Dart UTF-16 offset splits a surrogate pair"
            );
            if ch == '\n' {
                line = line.checked_add(1).context("Dart line count overflow")?;
            }
            scanned += 1;
            if scanned.is_multiple_of(4096) {
                check_deadline()?;
            }
        }
        anyhow::ensure!(
            line == expected_line,
            "Dart reference location/source mismatch"
        );
        lines.push(line);
    }
    Ok(lines)
}

struct Snapshot {
    sources: HashMap<String, u64>,
    context: HashMap<PathBuf, Option<u64>>,
}

impl Snapshot {
    fn capture(ctx: &ChannelContext<'_>, root: &Path) -> Result<Self> {
        let manifest =
            crate::index::manifest::Manifest::load(&crate::util::config::manifest_path(ctx.root))?;
        let sources = Self::sources(ctx, root, &|| Ok(()))?;
        let indexed: HashMap<_, _> = manifest
            .files
            .into_iter()
            .filter(|(p, _)| p.ends_with(".dart"))
            .collect();
        anyhow::ensure!(
            sources == indexed,
            "Dart index source hashes are stale; run `vex update` before strict lookup"
        );
        Ok(Self {
            sources,
            context: Self::context(root, &|| Ok(()))?,
        })
    }

    fn sources(
        ctx: &ChannelContext<'_>,
        root: &Path,
        check_deadline: &impl Fn() -> Result<()>,
    ) -> Result<HashMap<String, u64>> {
        crate::util::walk::discover_source_files(root, ctx.excludes)?
            .into_iter()
            .filter(|(_, lang)| *lang == Language::Dart)
            .filter(|(path, _)| std::fs::metadata(path).is_ok_and(|meta| meta.len() <= 1_048_576))
            .map(|(path, _)| {
                check_deadline()?;
                Ok((
                    to_rel_posix(&path, root).context("Dart file outside project root")?,
                    hash_file(&path)?,
                ))
            })
            .collect()
    }

    fn context(
        root: &Path,
        check_deadline: &impl Fn() -> Result<()>,
    ) -> Result<HashMap<PathBuf, Option<u64>>> {
        let mut paths = BTreeSet::new();
        for (file, lang) in crate::util::walk::discover_source_files(root, &[])? {
            check_deadline()?;
            if lang != Language::Dart {
                continue;
            }
            if let Some(parent) = file.parent() {
                for directory in parent.ancestors() {
                    for name in [
                        "pubspec.yaml",
                        "pubspec.lock",
                        "analysis_options.yaml",
                        ".dart_tool/package_config.json",
                        ".dart_tool/package_graph.json",
                        ".fvmrc",
                    ] {
                        paths.insert(directory.join(name));
                    }
                    if directory == root {
                        break;
                    }
                }
            }
        }
        for directory in root.ancestors() {
            for name in [
                "pubspec.yaml",
                "analysis_options.yaml",
                ".dart_tool/package_config.json",
                ".fvmrc",
            ] {
                paths.insert(directory.join(name));
            }
        }
        paths
            .into_iter()
            .map(|path| {
                check_deadline()?;
                let hash = if path.exists() {
                    Some(hash_file(&path)?)
                } else {
                    None
                };
                Ok((path, hash))
            })
            .collect()
    }

    fn verify(
        &self,
        ctx: &ChannelContext<'_>,
        root: &Path,
        check_deadline: &impl Fn() -> Result<()>,
    ) -> Result<()> {
        check_deadline()?;
        anyhow::ensure!(
            Self::sources(ctx, root, check_deadline)? == self.sources
                && Self::context(root, check_deadline)? == self.context,
            "Dart source or project context changed during strict query; retry after `vex update`"
        );
        check_deadline()?;
        Ok(())
    }
}

pub(super) fn output(
    mut hits: Vec<HitLocation>,
    ctx: &ChannelContext<'_>,
    backend: Option<Value>,
) -> ChannelOutput {
    let pre_filter_count = hits.len();
    hits.retain(|hit| ctx.path_scope.accept(&hit.path));
    let dropped = DropCounts {
        scope: pre_filter_count - hits.len(),
        ..DropCounts::default()
    };
    ChannelOutput {
        available: true,
        unavailable_reason: None,
        pre_filter_count,
        hits,
        dropped,
        backend,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf16_offsets_preserve_unicode_and_reject_partial_surrogates() {
        let source = "// 🦊\r\nvoid café() {}";
        let mut offsets = [(13, 2), (0, 1), (13, 2), (20, 2)];
        assert_eq!(
            reference_lines(source, &mut offsets, &|| Ok(())).unwrap(),
            [1, 2, 2, 2]
        );
        assert!(reference_lines(source, &mut [(4, 1)], &|| Ok(())).is_err());
        assert!(reference_lines(source, &mut [(999, 2)], &|| Ok(())).is_err());
        assert!(reference_lines(source, &mut [(13, 1)], &|| Ok(())).is_err());
    }

    #[test]
    fn reference_conversion_checks_deadline_during_long_source_scan() {
        let source = "x".repeat(20_000);
        let calls = std::cell::Cell::new(0);
        let check = || {
            calls.set(calls.get() + 1);
            anyhow::ensure!(calls.get() < 4, "timeout");
            Ok(())
        };
        assert!(reference_lines(&source, &mut [(20_000, 1)], &check)
            .unwrap_err()
            .to_string()
            .contains("timeout"));
        assert_eq!(calls.get(), 4);
    }

    #[test]
    fn accessor_normalization_requires_exact_property_name_file_kind_and_synthetic_location() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("lib.dart"), "").unwrap();
        std::fs::write(root.path().join("other.dart"), "").unwrap();
        let file = normal_path(&root.path().join("lib.dart")).unwrap();
        let declaration = json!({"kind":"GETTER", "name":"value", "location":{"file":file,"offset":50,"length":5,"startLine":3,"startColumn":11}});
        let property = json!({"kind":"FIELD", "name":"value", "location":{"file":file,"offset":0,"length":5,"startLine":1,"startColumn":1}});
        ensure_resolved_declaration(&property, &declaration, &file).unwrap();
        for (key, value) in [("kind", json!("METHOD")), ("name", json!("other"))] {
            let mut wrong = property.clone();
            wrong[key] = value;
            assert!(ensure_resolved_declaration(&wrong, &declaration, &file).is_err());
        }
        for (key, value) in [
            ("offset", json!(51)),
            ("length", json!(0)),
            ("startLine", json!(3)),
            ("startColumn", json!(11)),
            ("file", json!(root.path().join("other.dart"))),
        ] {
            let mut wrong = property.clone();
            wrong["location"][key] = value;
            assert!(ensure_resolved_declaration(&wrong, &declaration, &file).is_err());
        }
        let mut method = declaration.clone();
        method["kind"] = json!("METHOD");
        assert!(ensure_resolved_declaration(&property, &method, &file).is_err());
        let mut setter = declaration.clone();
        setter["kind"] = json!("SETTER");
        let mut variable = property.clone();
        variable["kind"] = json!("TOP_LEVEL_VARIABLE");
        ensure_resolved_declaration(&variable, &setter, &file).unwrap();
        ensure_resolved_declaration(&declaration, &declaration, &file).unwrap();
    }

    #[test]
    fn real_server_resolves_exact_library_receiver_private_and_constructor_identity() {
        let Some(sdk) = std::env::var_os("VEX_DART_SDK") else {
            eprintln!("SKIP: VEX_DART_SDK absent; real compiler test requires installed SDK");
            return;
        };
        let sdk = PathBuf::from(sdk);
        let dart = if sdk.is_dir() {
            sdk.join("bin")
                .join(if cfg!(windows) { "dart.exe" } else { "dart" })
        } else {
            sdk
        };
        let tmp = tempfile::tempdir().unwrap();
        let root = normal_path(tmp.path()).unwrap();
        std::fs::create_dir(root.join("lib")).unwrap();
        std::fs::create_dir(root.join(".dart_tool")).unwrap();
        std::fs::write(
            root.join("pubspec.yaml"),
            "name: fixture\nenvironment:\n  sdk: '>=3.0.0 <4.0.0'\n",
        )
        .unwrap();
        std::fs::write(root.join(".dart_tool/package_config.json"), r#"{"configVersion":2,"packages":[{"name":"fixture","rootUri":"../","packageUri":"lib/","languageVersion":"3.0"}]}"#).unwrap();
        std::fs::write(root.join("lib/a.dart"), "library a;\npart 'part.dart';\nvoid helper() {}\nvoid _private() {}\nclass Service {\n  Service();\n  Service.named();\n  void run() {}\n}\nvoid own() { _private(); }\n").unwrap();
        std::fs::write(root.join("lib/b.dart"), "void helper() {}\nvoid _private() {}\nclass Service {\n  Service();\n  void run() {}\n}\n").unwrap();
        std::fs::write(
            root.join("lib/part.dart"),
            "part of a;\nvoid partCaller() { helper(); _private(); }\n",
        )
        .unwrap();
        std::fs::write(root.join("lib/main.dart"), "import 'package:fixture/a.dart' as aa;\nimport 'b.dart' as bb;\nvoid caller() {\n  aa.helper();\n  bb.helper();\n  final x = aa.Service.named();\n  final y = bb.Service();\n  x.run();\n  y.run();\n  dynamic d = x; d.run();\n  final helper = () {}; helper();\n  final emoji = '🦊'; aa.helper();\n  aa.Service();\n}\n").unwrap();
        let a = normal_path(&root.join("lib/a.dart")).unwrap();
        let b = normal_path(&root.join("lib/b.dart")).unwrap();
        let mut server = Server::start(&dart, &root.join("server-cache")).unwrap();
        server.analyze(&root, &[a.clone(), b.clone()], &[]).unwrap();
        let cases = [
            (
                &a,
                "helper",
                3,
                "FUNCTION",
                vec![("main.dart", 4), ("main.dart", 12), ("part.dart", 2)],
            ),
            (&b, "helper", 1, "FUNCTION", vec![("main.dart", 5)]),
            (&a, "run", 8, "METHOD", vec![("main.dart", 8)]),
            (&b, "run", 5, "METHOD", vec![("main.dart", 9)]),
            (
                &a,
                "_private",
                4,
                "FUNCTION",
                vec![("a.dart", 10), ("part.dart", 2)],
            ),
            (&b, "_private", 2, "FUNCTION", vec![]),
            (
                &a,
                "Service.named",
                7,
                "CONSTRUCTOR",
                vec![("main.dart", 6)],
            ),
            (&a, "Service", 6, "CONSTRUCTOR", vec![("main.dart", 13)]),
            (&b, "Service", 4, "CONSTRUCTOR", vec![("main.dart", 7)]),
        ];
        let mut requests = Vec::new();
        let mut declarations = Vec::new();
        for (file, name, line, kind, _) in &cases {
            let mut elements = Vec::new();
            collect_elements(server.outline(file).unwrap(), &mut elements);
            let element = elements
                .into_iter()
                .find(|e| {
                    e["name"] == *name && e["kind"] == *kind && e["location"]["startLine"] == *line
                })
                .unwrap()
                .clone();
            requests.push((
                (*file).clone(),
                element["location"]["offset"].as_u64().unwrap(),
            ));
            declarations.push(element);
        }
        let results = server.references_many(&requests).unwrap();
        assert_eq!(results.len(), cases.len());
        for (((file, name, line, _, expected), element), (resolved, refs)) in
            cases.into_iter().zip(declarations).zip(results)
        {
            assert_eq!(
                resolved["location"], element["location"],
                "wrong identity for {file:?}:{line}"
            );
            let mut actual: Vec<_> = refs
                .iter()
                .map(|r| {
                    assert_eq!(
                        r["isPotential"], false,
                        "dynamic receiver must not produce confirmed references"
                    );
                    let location = &r["location"];
                    (
                        Path::new(location["file"].as_str().unwrap())
                            .file_name()
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .to_owned(),
                        location["startLine"].as_u64().unwrap(),
                    )
                })
                .collect();
            actual.sort();
            let mut expected: Vec<_> = expected
                .into_iter()
                .map(|(p, l)| (p.to_owned(), l))
                .collect();
            expected.sort();
            assert_eq!(
                actual, expected,
                "wrong resolved refs for {file:?}:{line} {name}"
            );
        }
    }
}
