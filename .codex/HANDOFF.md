# Agent Handoff

## Current task

Design the first Rust custom IR and its model-facing structural actions.

Current working files:

- `src/model/tokens.md`: model-visible actions grouped by the current `Need`
- `src/model/ir.rs`: completed in-memory semantic/source IR
- `spec.md`: project specification

Keep model actions separate from in-memory containers. Enum wrappers and deterministic fields in `ir.rs` do not automatically require model tokens.

## Current direction

- Separate Rust statements from expressions according to where they are legal and composable.
- Keep block generation token-efficient. Under `Need<BlockEntryOrEnd>`, the model selects another block entry or one of two end actions: `EndAndYield` and `EndAndDiscard`. Do not add a `Continue` action.
- A completed expression can remain pending in the controller. Starting another entry commits it as an expression statement. `EndAndYield` makes it the block tail; `EndAndDiscard` commits/discards it and ends the block with unit.
- `EndAndYield` is valid only when a completed expression is available. `EndAndDiscard` is valid for an empty block and after statements or expressions.
- Use dedicated structs to hold fields for compound enum variants, but let the controller create deterministic wrappers and fields without extra learned actions.
- Keep completed IR separate from controller construction state. Prefer empty `Vec<T>` over `Option<Vec<T>>`; reserve `Option<T>` for meaningful source-level absence.

## Open design work

- Finalize the exact containment structs for expressions, statements, blocks, calls, closures, and types in `src/model/ir.rs`.
- Decide whether the MVP supports only direct calls by `SymbolId`, or also paths and expression-valued callees. Direct-symbol calls are acceptable for the MVP.
- Define closure parameters and source-level `move` capture without requiring compiler inference.
- Start closure parameters as simple bindings if needed; introduce a `Pattern` IR when destructuring is supported.
- Decide whether block-local item declarations are in the MVP. If not, omit `Statement::Item` initially.
- Expand compound type variants so references, pointers, slices, tuples, and custom generics retain their component data.
- Define which inferred types and closure capture modes are optional compiler annotations rather than base source IR.

## Validation after IR edits

1. Run `cargo check`.
2. Confirm recursive IR cycles contain an indirection such as `Box` or `Vec`.
3. Confirm empty collections have one canonical representation.
4. Verify action traces distinguish `{ expr }` from `{ expr; }` through the two block-end actions.
5. Update this handoff with each settled decision and remaining question.

## Collaboration notes

Record future agent decisions and unresolved issues in this file. Keep entries short and include the affected file and section. Update this handoff during each substantive task, not only at final handoff.

### 2026-09-06: Initial IR and block actions

Affected files: `src/model/tokens.md`, `src/model/ir.rs`.

- `Need<BlockEntryOrEnd>` offers block entries plus `EndAndYield` and `EndAndDiscard`.
- The user explicitly rejected a separate `Continue` action because selecting another entry already continues the block.
- Statements and expressions are separate in memory, but deterministic containment wrappers must not cost model actions.
- Calls may use a direct `SymbolId` target in the MVP; arbitrary expression-valued callees are a later completeness feature.
- Closure parameters and closure captures are distinct. Parameters are provided by callers. Captures reference symbols from enclosing scopes.
- Preserve the explicit closure `move` keyword in base IR. Precise inferred capture modes belong in optional compiler metadata.
- Parameter binding mutability, reference mutability, and ownership are distinct. Explicit ownership/borrowing is represented by type syntax; omitted closure parameter types remain unresolved until optional compiler analysis.
- Patterns generalize bindings and destructuring. A simple `SymbolId` binding is sufficient until tuple, enum, struct, wildcard, or reference patterns enter scope.
- `Statement::Item` means a declaration nested in a block. It can be omitted from the MVP if local item declarations are unsupported.
- Primitive type variants are leaves. Compound type variants must retain components, such as the referent of `&T`, tuple members, and slice element type.

### 2026-08-29: English conditioning

Updated `more_patterns_per_pattern_spec_condensed.md`, Sections 1, 10, 13, 17, 21, 25, 27, and 29.

- The controller feeds the current `Need`, state, and partial-program context to the structural decoder.
- A separate English encoder is optional and does not receive controller feedback.
- The preferred first baseline is decoder-only prompt conditioning with separate text and structural-action embeddings.
- The encoder-decoder design remains an equal-compute comparison.
- Standard Transformer blocks can be reused. Structural inputs, constrained heads, symbol pointers, masks, and the controller interface remain custom.
