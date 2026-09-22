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
- Symbols have stable `SymbolId` identities. Deferred two-phase naming through
  `NameRegistry` (register the complete context, then seal and optionally assign
  names) is the current default; callers choose the registration order and
  symbol universe. That universe includes declarations plus ambient symbols
  referenced by block fragments. Every registered symbol reserves its fallback,
  and the renderer borrows the sealed registry as its single naming source.
- Rendering checks syntax and naming consistency only. The controller/context
  owns semantic visibility and call-legality validation.
- Integration of the future English-token naming head remains future work.

## Structural action-trace milestone

- `src/controller.rs` deterministically derives and validates typed action traces
  from completed MVP IR. It uses declaration allocations, symbol-pointer
  actions, list continuation/end actions, and the two settled block endings.
- This is trace generation only. A mutable controller that applies/replays
  actions and allocates symbols during decoding remains future work.
