# Dart indexing

Vex registers `.dart` as language ID 20 using
[`tree-sitter-dart-orchard` 0.7.0](https://crates.io/crates/tree-sitter-dart-orchard/0.7.0).
The exact pin is reproducible; it does not establish grammar superiority or full
language coverage. The [crate manifest](https://codeberg.org/grammar-orchard/tree-sitter-dart-orchard)
names Tim Whiting as author, and the [crates.io version record](https://crates.io/api/v1/crates/tree-sitter-dart-orchard/0.7.0)
lists Antonin Delpeuch (`wetneb`) as publisher. Orchard release metadata names
[`cfping/tree-sitter-dart`](https://github.com/cfping/tree-sitter-dart) as upstream.
These sources do not establish Orchard's current maintainer, which remains
unverified. Orchard exposes a `LanguageFn` compatible with this project's
Tree-sitter 0.26 dependency; cfping's [current manifest](https://github.com/cfping/tree-sitter-dart/blob/master/Cargo.toml)
uses Tree-sitter 0.25.3. This is a concrete API compatibility rationale for
the published crate, not a comparative quality claim.

The index extracts classes, enums, mixins, named extensions, extension types,
functions, methods, constructors, type aliases, enum constants, and import
URIs. Calls are syntactic name edges; hierarchy edges preserve `extends`,
`implements`, and mixin `with` relations. References use the shared
`is_meaningful_identifier` filter and omit comments and plain string text while
retaining identifiers in interpolation. Imports and references are not
compiler-resolved.

Fixtures cover selected declarations, calls, relations, aliases, imports,
interpolation, and test-path classification. The [record syntax fixture](../tests/dart_query_test.rs)
parses a record type, literal, and destructuring pattern without Tree-sitter
errors. It does not verify all [record](https://dart.dev/language/records) or
pattern forms. The fixtures do not establish complete Dart 3 compatibility.
Orchard can recover from unsupported syntax with partial trees. Class modifiers,
switch forms, and other modern syntax remain unverified without dedicated
fixtures. Earlier parser comparisons observed recovery on valid constructs,
including higher-order and null-asserted invocations. A successful parse or
index does not prove complete syntax support.

Indexing is offline and does not execute or require the Dart SDK. No strict
compiler-backed Dart references are provided. See
[`SUPPORTED_LANGUAGES.md`](SUPPORTED_LANGUAGES.md) for the support matrix.
