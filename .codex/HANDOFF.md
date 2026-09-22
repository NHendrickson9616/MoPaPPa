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

## Structural action-trace milestone

- `src/controller.rs` deterministically derives and validates typed action traces
  from completed MVP IR. It uses declaration allocations, symbol-pointer
  actions, list continuation/end actions, and the two settled block endings.
- This is trace generation only. A mutable controller that applies/replays
  actions and allocates symbols during decoding remains future work.
