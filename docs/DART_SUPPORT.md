# Dart support

This branch registers `.dart` as language ID20 without changing existing IDs.
The grammar is pinned to `tree-sitter-dart-orchard =0.7.0`.

## Extraction

- Classes, enums, mixins, named extensions and extension types are indexed.
- Top-level functions, methods, getters/setters and constructors use the existing
  Vex symbol schema. Callable body tokens include the declaration body rather
  than only the nested signature.
- Import sites retain their literal URI; they are not compiler-resolved imports.
- Call queries select invocation nodes. Bare identifier reads, comments and
  literal string text do not create call edges.
- Normal, member, conditional, cascade and generic constructor-like calls use
  the existing innermost named-function ownership and name-based resolver.
- Hierarchy extraction preserves `extends`, `implements` and mixin `uses` edges.
- AST refs skip comments and retain code inside string interpolation.
- Pattern skeletons register Dart declaration nodes. ast-grep remains the chosen
  structural-search route on hosts that apply the shared code-tooling policy.

## Boundaries

Dart does not have a Vex scope binder. `usages --strict`, imported-name identity
and receiver/type resolution are not supported by this addition. Ordinary refs
and call graph resolution retain Vex's name-based behavior and its ambiguity
limits. Anonymous callbacks attach to the enclosing named callable.

Orchard0.7.0 can recover from unsupported syntax and produce partial trees. A
successful parse or index does not guarantee complete Dart3.13 syntax coverage.
The earlier vida_mobile parser comparison observed ERROR/MISSING recovery on
valid Dart source, including higher-order and null-asserted invocations. This
branch integrates the published grammar; it does not rewrite that grammar.

## Verification

`cargo test --test dart_query_test` covers loading, symbols/imports, ownership,
generic/conditional/cascade calls, non-call identifier filtering, body tokens,
type aliases, hierarchy kinds and string interpolation.

The language is also included in shared-tree, parser-count, query-compilation
and snapshot coverage. On Windows the release uses `--features gpu-directml`
to preserve the installed GPU capability.
