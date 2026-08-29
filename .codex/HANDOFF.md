# Agent Handoff

## Current task

The user requested a shorter, less formulaic version of `more_patterns_per_pattern_spec.md`. The user also requested corrected section numbering.

## File policy

- Keep `more_patterns_per_pattern_spec.md` unchanged as the original draft.
- Use `more_patterns_per_pattern_spec_condensed.md` for the edited version.
- Do not replace or delete either file without an explicit request.

## Editing choices

The condensed version:

- uses sequential section and subsection numbers
- merges repeated architecture, controller, symbol, training, and prototype sections
- preserves `DECIDED`, `LIKELY`, and `OPEN` labels
- uses `custom IR`, `controller`, `structural decoder`, and `structural action` consistently
- preserves caveats about partial code, compiler context, macros, naming, and untested efficiency claims
- removes informal notes and the unrelated closing quotation

## Validation to repeat after edits

1. Check that top-level headings form one sequence without gaps.
2. Check that subsection prefixes match their parent sections.
3. Confirm that the original draft has no worktree diff caused by this task.
4. Review Markdown diagnostics if the editor provides them.

## Collaboration notes

Record future agent decisions and unresolved issues in this file. Keep entries short and include the affected file and section.

### 2026-08-29: English conditioning

Updated `more_patterns_per_pattern_spec_condensed.md`, Sections 1, 10, 13, 17, 21, 25, 27, and 29.

- The controller feeds the current `Need`, state, and partial-program context to the structural decoder.
- A separate English encoder is optional and does not receive controller feedback.
- The preferred first baseline is decoder-only prompt conditioning with separate text and structural-action embeddings.
- The encoder-decoder design remains an equal-compute comparison.
- Standard Transformer blocks can be reused. Structural inputs, constrained heads, symbol pointers, masks, and the controller interface remain custom.
