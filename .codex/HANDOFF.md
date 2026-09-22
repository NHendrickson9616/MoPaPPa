# Agent Handoff

## Executable MVP vertical slice

- The initial implementation is a public structural Rust IR in
  `src/model/ir.rs` and a deterministic renderer in `src/renderer.rs`; this
  comes before controller/model work.
- `Root::Module` renders top-level items (currently free functions), while
  `Root::BlockFragment` renders a syntactically valid block expression.
- MVP nodes: functions, primitive types, blocks, `let`, symbol references,
  integer/string/bool literals, binary expressions, direct `SymbolId` calls,
  and `if`.
- Blocks represent semicolon-terminated expression statements separately from
  an optional tail expression.
- Closures, patterns, local items, `match`, loops, paths/methods, and compiler
  annotations are intentionally out of scope.
- Symbols have stable `SymbolId` identities. Rendering receives their names
  separately, rejects invalid name-map entries, and uses deterministic valid
  fallback identifiers when unmapped. Rendered symbols must resolve to distinct
  spellings, preventing accidental aliases.
