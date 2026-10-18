# Dart indexing

Vex pins `tree-sitter-dart-orchard` 0.7.0 for `.dart` files. Indexing is
offline and does not execute or require the Dart SDK.

The extractor indexes classes, enums, mixins, extensions, extension types,
functions, methods, constructors, type aliases, enum constants, and imports.
Constructors use qualified identities such as `Repo.new` and `Repo.named`.
Class static methods use `Owner.member`; top-level functions retain their
declared names, including names that start with an uppercase letter.
Call edges are syntactic; hierarchy edges retain `extends`, `implements`, and
`with` relations. References use the shared identifier filter and are not
resolved by a compiler or across imported libraries.

Fixtures cover selected declarations, constructors, calls, hierarchy
relations, aliases, imports, interpolation, and test paths. They do not prove
complete Dart 3 syntax coverage. Unsupported syntax may produce partial trees;
records, patterns, class modifiers, and other modern forms remain incompletely
verified. There is no strict compiler-backed Dart reference binder.

See [Supported languages](SUPPORTED_LANGUAGES.md) for the support matrix.
