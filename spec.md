# More Patterns Per Pattern

# 1. Project Overview

- **DECIDED**: Generate Rust through a structured custom IR instead of source-code tokens.
- **DECIDED**: A deterministic renderer must convert the custom IR into normal Rust source.
- **DECIDED**: The custom IR must support expressions, items, modules, partial code, and code with unresolved context.
- **DECIDED**: The controller should reconstruct syntax that does not require a learned decision.
- **DECIDED**: Internal symbol identity should not depend on identifier spelling.
- **DECIDED**: The structural decoder receives the current `Need`, controller state, and partial program.
- **LIKELY**: The first English-conditioned baseline will use the prompt as a decoder-only prefix.
- **OPEN**: Compare that baseline with a separate pretrained English encoder.
- **LIKELY**: A separate naming pass will assign readable identifiers.

The custom IR must preserve enough source-level structure to produce idiomatic Rust. Reversibility does not require identical formatting or punctuation.

---

# 2. Motivation

Programming stenography separates intended words from their letter-by-letter spelling. This project applies a similar split to code generation.

The model predicts program structure. Deterministic software supplies grammar, punctuation, and formatting where possible.

This split may let the model spend more capacity on program behavior and relationships. Experiments must test that claim.

---

# 3. Core Hypothesis

Conventional code generation learns both program design and source serialization. Source serialization includes braces, commas, semicolons, precedence parentheses, indentation, and other required syntax.

The proposed system moves predictable serialization into a deterministic controller and renderer.

## 3.1 Structural decoder responsibilities

The structural decoder predicts:

- item and expression kinds
- operations and types
- symbol creation and references
- variable-length list termination
- source-level choices that affect readable Rust
- semantic relationships available from context

## 3.2 Controller and renderer responsibilities

The controller and renderer manage:

- the custom IR schema
- valid structural actions
- field order and node completion
- punctuation and delimiters
- precedence parentheses
- formatting
- scope and internal symbol tables

## 3.3 Conceptual layers

The design separates four layers:

1. **Intent**: The English request.
2. **Program structure**: Functions, calls, loops, types, borrows, and other custom IR nodes.
3. **Program identity**: Internal symbol IDs such as `@0` and `@1`.
4. **Source text**: Names, punctuation, formatting, and rendered Rust.

The structural decoder primarily handles program structure. The controller handles bookkeeping, and the naming pass handles readable identifiers.

## 3.4 Research hypothesis

The design may improve:

- structural validity
- sample efficiency
- long-range consistency
- semantic density per learned decision
- capacity available for higher-level program patterns

These are hypotheses, not established results.

---

# 4. Related Code-Model Work

Relevant research areas include:

- **ChainCoder / Outline, Then Details**: AST-guided coarse-to-fine generation.
- **AST-T5**: AST-aware segmentation and masking while retaining source or subword output.
- **GraphCodeBERT**: Data-flow-aware code representations.
- **CodeT5**: Identifier-aware code modeling.
- **TokDrift**: Mismatch between source-equivalent code and subword tokenization.
- **CodeBPE**: Code-specific statistical tokenization.
- **LangMAP**: Language-adaptive tokenization with improved code-boundary alignment.
- **SemToken**: Semantic-aware tokenization.
- **ChopChop**: Semantically constrained program generation.
- **Latent-DARM**: Latent communication between specialized neural components.
- Compiler IR studies, symbolic planning, formalization models, and shared-state systems are also relevant.

AST-native generation is not standard for several reasons:

- Text tokenizers support many languages and mixed prose or code formats.
- Text models tolerate malformed and partial code.
- ASTs do not provide type, borrow, data-flow, trait, or name resolution by themselves.
- Poor tree serializations can exceed source length.
- Large Transformers can learn much grammar from text.

This project must show an advantage over strong text-token baselines at equal compute.

---

# 5. Rust Representation Levels

An approximate Rust compiler pipeline is:

```text
Rust source
    ↓
AST
    ↓
HIR
    ↓
THIR
    ↓
MIR
    ↓
LLVM IR
    ↓
machine code
```

This is a conceptual overview, not a stable rustc interface.

The AST represents written syntax. It naturally preserves `if let`, `for`, calls, blocks, patterns, and type syntax.

It works well for incomplete snippets but contains little resolved semantic information.

HIR normalizes high-level Rust and removes syntax made redundant by tree structure. It is less noisy than an AST but can erase choices needed for idiomatic source reconstruction.

THIR adds types and resolved operations. It requires context that isolated snippets often lack.

MIR makes control flow, moves, borrows, drops, and calls explicit. It is useful for compiler analysis but too low-level for easy reconstruction of idiomatic Rust.

- **LIKELY**: Do not copy rustc HIR directly.
- **DECIDED**: Build a custom, reversible semantic surface IR.

---

# 6. Custom IR and Reversibility

The required pipeline is:

```text
Rust source
    ↓
custom IR
    ↓
deterministic renderer
    ↓
Rust source
```

The renderer can normalize formatting. It can infer:

- braces and parentheses
- commas and most semicolons
- whitespace and indentation
- `->`
- generic brackets
- punctuation implied by node kinds

For example:

```rust
fn foo(x: &mut Vec<i32>) -> Option<i32>
```

can use this structure:

```text
FUNCTION
├── PARAM
│   └── MUT_REF
│       └── GENERIC_TYPE Vec
│           └── I32
└── RETURN
    └── GENERIC_TYPE Option
        └── I32
```

The renderer supplies `fn`, delimiters, `:`, `&mut`, `->`, commas, and spacing.

Round-trip tests must define equivalence at the custom IR level. Rendered code can differ in formatting while preserving the same represented program.

---

# 7. Source-Level Constructs

- **DECIDED**: Preserve a source construct when it affects readable or idiomatic Rust.

Likely first-class nodes include:

- `FOR_LOOP`, `WHILE_LOOP`, `WHILE_LET`, and `LOOP`
- `IF`, `IF_LET`, and `MATCH`
- `TRY_EXPR` and `AWAIT`
- `CLOSURE` and `ASYNC_BLOCK`
- `METHOD_CALL` and `RANGE`
- patterns, references, and mutability
- generics, visibility, and attributes

Normalization must not remove distinctions required for source reconstruction.

---

# 8. Optional Compiler Annotations

The base custom IR must remain usable when semantic resolution fails.

```text
METHOD_CALL
├── receiver = @3
├── method = NAME_SLOT
└── args = [...]
```

When compiler context exists, annotations can add:

```text
receiver_type = Vec<T>
resolved_method = Vec::push
return_type = ()
trait_resolution = ...
```

- **DECIDED**: Compiler data enriches the custom IR but is not required for inclusion.

This rule supports both isolated snippets and complete crates.

---

# 9. Initial Compiler Study

- **LIKELY**: First study rustc AST-to-HIR lowering across a representative construct corpus.

The study should identify:

- syntax that rustc removes
- syntax that rustc normalizes
- complete desugarings
- simplifications useful to this project
- transformations that prevent source reconstruction

Adopt useful structural simplifications. Preserve distinctions required for reversible rendering.

---

# 10. Structural Decoding Architecture

The model does not serialize a complete tree directly. It works with a deterministic controller:

1. The controller identifies the required field.
2. The structural decoder scores structural actions.
3. The controller masks invalid actions.
4. The controller applies the selected action to the custom IR.
5. The controller advances to the next field.
6. The process repeats until the root is complete.

## 10.1 Structural decoder

The structural decoder receives the English condition, previous structural actions, current `Need`, controller state, and partial-program context. It produces hidden states and action scores.

Its attention and feed-forward blocks can use standard Transformer implementations. Its structural inputs, output heads, pointer mechanism, and controller interface are custom.

The decoder does not contain an explicit Rust grammar.

## 10.2 Controller

The controller knows:

- the custom IR schema
- the current node and field
- the expected action category
- scopes and symbols
- valid next actions
- node completion rules

The controller masks invalid actions, updates the custom IR, and maintains traversal state. At each step, it returns the next `Need`, state, and partial-program context to the decoder.

## 10.3 Feedback loop

```text
English condition ──────────────────────────────┐
                                                ▼
Previous actions + current Need + state ──► structural decoder
                                                │
                                                ▼
                                           action scores
                                                │
                                                ▼
                                         controller mask
                                                │
                                                ▼
                                          chosen action
                                                │
                                                ▼
                                            controller
                                                │
                           update custom IR, Need, and state
                                                │
                                                └────────────► structural decoder
```

---

# 11. Grammar Masking

Suppose the current requirement is:

```text
Need(Item)
```

Valid actions include `Function`, `Struct`, `Enum`, `Trait`, and `Impl`. Actions such as `If`, `Call`, and `I32` are invalid in this field.

The controller sets invalid action scores to negative infinity before softmax. Invalid actions then receive zero probability.

The model still learns which valid action fits the request. It does not need to learn every grammar prohibition from examples.

---

# 12. Schema Traversal and Completion

The custom IR schema defines node fields:

```rust
Function {
    name: Name,
    params: Vec<Param>,
    return_type: Option<Type>,
    body: Block,
}
```

Selecting `Function` creates these fields. The model does not predict that the fields exist.

A `BinaryOp` has an operator, a left expression, and a right expression. The controller completes the node after all three fields are present.

Blocks, parameter lists, match arms, generic lists, module items, and tuples have variable length. Their termination remains a learned fact.

Structural decoding removes the need to predict a literal closing delimiter. It does not remove the decision that a list has ended.

When a child node completes, the controller marks its parent field complete. It closes a fixed-shape parent after all required fields are complete.

---

# 13. Decoder State and Context

The structural decoder must know what to predict and what the program already contains.

Its inputs are approximately:

```text
English condition
+
previous structural actions or custom IR built so far
+
current Need
+
current controller state
```

The English condition can be prompt-prefix embeddings or hidden states from a separate encoder. The changing `Need` and controller state always enter the structural decoder, not the optional English encoder.

## 13.1 Compositional state embeddings

A state can contain:

```text
parent = FUNCTION
field  = RETURN_TYPE
Need   = TYPE
```

The `Need` selects the required action category. The remaining state identifies the location and relevant controller context.

The model can combine learned embeddings:

```text
S = E_parent(FUNCTION)
  + E_field(RETURN_TYPE)
  + E_expected(TYPE)
```

This representation reuses concepts such as `TYPE` across many fields.

## 13.2 Program-so-far encoding

Possible encodings include:

- a linearized list of constructed nodes
- a tree encoder
- attention over node embeddings
- persistent decoder state with explicit tree context
- graph or tree positional encodings

The encoding must retain declarations and symbol uses without recreating excessive serialization overhead.

- **OPEN**: Choose the exact state and partial-tree encoding.

---

# 14. Specialized Prediction Heads

The model does not need one universal action vocabulary.

| Requirement | Prediction space |
|---|---|
| `Need<Item>` | Function, Struct, Enum, Trait, Impl |
| `Need<Expression>` | Call, BinaryOp, If, Match, Literal, SymbolRef |
| `Need<Type>` | I32, Bool, Ref, Tuple, GenericType |
| `Need<BinaryOperator>` | Add, Sub, Mul, Div, Eq |
| `Need<ListContinue>` | Yes, No |
| `Need<new binding>` | New symbol allocation |
| `Need<SymbolRef.target>` | Pointer over available symbols |
| `Need<literal value>` | Literal-specific mechanism |

Specialized heads reduce irrelevant competition and support dynamic selection mechanisms.

- **OPEN**: Compare specialized heads with a universal action head.

---

# 15. Symbols, Scope, and References

## 15.1 Symbol creation

When a declaration needs an identity, the model predicts `NEW_SYMBOL`. The controller allocates an internal ID such as `@0` or `@1`.

```text
Need<Function.name> → NEW_SYMBOL → @0
Need<Param.binding> → NEW_SYMBOL → @1
```

## 15.2 Symbol references

For a reference, the controller exposes symbols that are legal in lexical scope. The model scores this dynamic candidate set.

This is pointer selection, not vocabulary growth.

Type, ownership, and borrow legality require semantic metadata or later compiler checks. Lexical scope alone cannot guarantee them.

## 15.3 Symbol table

The controller tracks:

- unique symbol ID
- declaration kind
- lexical scope
- visibility when relevant
- type when known
- ownership and borrow metadata when known
- available reference candidates

## 15.4 Name invariance

These functions should share one structural representation:

```rust
fn increment(count: i32) -> i32 {
    count + 1
}
```

```rust
fn addOne(x: i32) -> i32 {
    x + 1
}
```

A canonical form can use `@1` for either parameter. This supports shadowing, alpha-renaming, and a separate naming pass.

---

# 16. Naming and Lexical Content

A naming pass can assign readable names after structural generation.

Inputs can include:

- the English request
- the completed custom IR
- symbol roles and known types
- every use of each symbol
- project style context

Output can map:

```text
@0 → process_file
@1 → path
@2 → contents
@3 → result
```

The naming pass sees the complete program before choosing names. This can provide more context than naming each binding at declaration time.

Internal binding names can use this late pass. External paths, APIs, fields, methods, macros, and resolution-sensitive identifiers may need earlier lexical decisions.

- **OPEN**: Decide which identifiers belong to structural generation and which belong to the naming pass.
- **OPEN**: Decide whether naming trains jointly or after structural training.

---

# 17. English Conditioning Interface

A separate English encoder is optional. The required property is that every structural step can use the English request.

## 17.1 Preferred first baseline

- **LIKELY**: Start with a decoder-only structural model.

```text
English prompt embeddings
+
previous structural action embeddings
+
current Need and controller-state embeddings
                 ↓
       decoder-only Transformer
                 ↓
         Need-specific head
```

English tokens and structural actions can use separate embedding tables with the same hidden width. The controller supplies the changing `Need`, state, masks, and partial-program context.

The output remains custom even when the Transformer blocks are standard. Fixed action categories use constrained heads, symbols use a dynamic pointer head, and lexical fields use specialized mechanisms.

## 17.2 Encoder-decoder alternative

A pretrained English encoder can process the prompt once:

```text
English tokens → English encoder → contextual prompt states
                                           ↓ cross-attention
                                  structural decoder
```

The encoder does not receive controller feedback. The custom decoder receives the state, `Need`, and partial program, then cross-attends to the fixed prompt states.

A separate encoder provides bidirectional prompt representations, independent model sizes, and a clean boundary between text and structure. It does not already understand the custom IR. Paired training must align prompt states with structural actions.

A learned projection can connect different encoder and decoder widths.

## 17.3 Decision

- **DECIDED**: Do not require a separate encoder in the core architecture.
- **LIKELY**: Use decoder-only conditioning for the first English-conditioned prototype.
- **OPEN**: Compare it with a pretrained encoder and custom cross-attentive decoder at similar compute.

---

# 18. Literal Handling

Some content does not fit a small structural vocabulary. Examples include integers, floats, strings, characters, byte strings, raw strings, and format strings.

Possible mechanisms include:

- specialized literal heads
- byte or subword generation inside literal fields
- copying from the English request
- canonical numeric forms

Structured generation does not need to remove text tokenization from inherently lexical fields.

- **OPEN**: Choose literal encodings and define any role for the naming model.

---

# 19. Macros and Unresolved Syntax

Rust macros can contain arbitrary token trees and domain-specific syntax. Expansion can also depend on external definitions.

A strict ban on custom macros would conflict with support for arbitrary snippets. The custom IR therefore needs a policy for unresolved macro syntax.

Possible approaches include:

- first-class nodes for selected standard macros
- structured nodes for well-known macro forms
- an opaque token-tree fallback for custom or unresolved macros
- expanded forms when reliable compiler context exists

- **OPEN**: Define the macro policy and its reversibility guarantees.

Comments, unresolved paths, and other lexical content may need a similar fallback.

---

# 20. Training Data and Validation Policy

Potential Rust instruction datasets include:

- `Convence/Rust-Coder`
- `Neloy262/rust_instruction_dataset`
- `Maverfrick/Rust_dataset`
- the RustForge personal Rust dataset
- Rust-filtered CodeAlpaca data

Reported sizes, licenses, and content quality must be checked before use.

Do not assume each row is a complete, compilable program. Rows can contain snippets, missing imports, external dependencies, obsolete APIs, generated errors, or explanatory prose.

Store structured validation results:

```text
parse_failure
missing_dependency
unresolved_symbol
type_error
partial_semantic_resolution
full_semantic_success
```

This metadata can later support repair, robustness, partial-context, and compiler-error training.

---

# 21. Training Strategy

## 21.1 Structural Rust pretraining

Convert Rust corpora into the custom IR. Train the structural decoder on actions, state transitions, references, list termination, types, and operators.

The goal is to learn Rust structure before English alignment.

## 21.2 English and custom IR alignment

Use paired descriptions from instruction datasets, documentation, function descriptions, tests, and sufficiently aligned issues or commits.

Train:

```text
English → custom IR
```

## 21.3 Synthetic description expansion

For real Rust code:

1. Derive the custom IR deterministically.
2. Use a strong model to generate descriptions.
3. Train English-to-custom-IR generation.

The target remains parser-derived. Only the language-side supervision is synthetic.

## 21.4 Joint fine-tuning

Train the structural decoder, text and action embeddings, state embeddings, symbol pointer, and possibly the naming model. If experiments add a separate encoder, first train its projection or cross-attention interface, then evaluate joint fine-tuning.

## 21.5 Compiler and test feedback

Use applicable parse, render, type, borrow, test, and semantic checks. Section 26 defines the evaluation metrics.

---

# 22. Structural Training Traces

Given a correct custom IR tree, the controller can produce training states deterministically.

```text
STATE                               TARGET

Need<Item>                          Function
Need<Function.name>                 NewSymbol
Need<Param.binding>                 NewSymbol
Need<Param.type>                    I32
Need<Params.binding>                EndParam
Need<Function.return_type>          I32
Need<Expression>                    BinaryOp
Need<BinaryOperator>                Mul
Need<Binary.left>                   SymbolRef
Need<SymbolRef.target>              @1
Need<Binary.right>                  IntLiteral
Need<IntLiteral.value>              2
Need<Expression>                    EndFunction
```

This is teacher forcing over structural actions instead of source tokens. It is also only 13 tokens.

---

# 23. Structural Correctness by Construction

The controller can prevent schema-level errors:

- A `BinaryOp` cannot have three operands.
- A type field cannot receive an `If` expression.
- A function cannot omit required fields.
- A fixed-shape node cannot remain unclosed.
- A reference can be limited to symbols in lexical scope.
- The model does not need to close source delimiters.

The controller does not guarantee type correctness, borrow correctness, algorithm choice, or semantic correctness.

---

# 24. Efficiency Hypothesis

The architecture removes learned predictions for repeated syntax such as:

```text
( ) { } , ; -> < >
```

It adds structural state transitions, list-termination actions, tree construction, and controller interactions. It can therefore require more neural steps than expected.

- **OPEN**: Measure efficiency rather than assuming an improvement.

Compare models of similar size and compute. Measure neural steps, training cost, validity, and task accuracy.

---

# 25. Main Open Questions

- What custom IR node set should exist?
- Which Rust constructs should remain explicit or become normalized?
- What equivalence should round-trip tests require?
- How should macros, comments, literals, unresolved names, imports, and paths be represented?
- Can part of the custom IR become language-independent without losing Rust-specific meaning?
- How should the decoder encode the partial custom IR?
- Should prediction use one action head or specialized heads?
- How should symbol pointer scoring work?
- Which identifiers can wait for the naming pass?
- How should optional type and compiler annotations appear?
- Does decoder-only prompt conditioning outperform a separate pretrained encoder at similar compute?
- Should the naming model train jointly or separately?
- Does structural decoding improve results at equal compute?

---

# 26. Evaluation

## 26.1 Structural metrics

- valid custom IR rate
- renderer success rate
- parser and custom IR round-trip rate
- grammar violation rate

## 26.2 Compiler metrics

When context permits, measure:

- parse success
- type-check success
- borrow-check success
- compile success

## 26.3 Semantic metrics

- unit-test pass rate
- functional equivalence
- benchmark task success
- HumanEval-style correctness where applicable

## 26.4 Representation and efficiency metrics

- source-token count versus structural-action count
- entropy per prediction
- tree depth
- learned decisions removed by the controller
- training FLOPs per valid program

All comparisons must use strong baselines at similar model size and compute.

---

# 27. Initial Prototype Plan

## 27.1 Define a small custom IR

Start with:

- functions and parameters
- primitive types
- blocks and `let`
- symbol references
- integer literals
- binary operations
- calls and `if`

Add loops after the basic pipeline works.

## 27.2 Build deterministic conversion

Implement:

```text
Rust source → parser or AST → custom IR
custom IR → Rust source → rustfmt
```

Define and test round-trip equivalence before adding machine learning.

## 27.3 Build the controller

Implement schema traversal, field completion, list termination, symbol allocation, scope tracking, and validity masks.

Use known trees to verify that action traces reconstruct the same custom IR.

## 27.4 Train the structural model

Train next-action prediction from the current state and partial program. Do not add English conditioning yet.

## 27.5 Add English conditioning

Use English prompt embeddings as a prefix to the structural action sequence. Keep separate embedding tables for text tokens and structural actions.

Feed the current `Need` and controller state into each structural step. Add a separate encoder only as a comparison after this baseline works.

## 27.6 Add symbols and naming

Add dynamic symbol pointer selection. Then train the naming pass over completed custom IR trees.

## 27.7 Scale and compare

Process larger datasets and repositories. Compare against a text-token baseline at similar model size and compute.

---

# 28. End-to-End Example

Source:

```rust
fn double(x: i32) -> i32 {
    x * 2
}
```

Custom IR:

```text
STATE                               TARGET

Need<Item>                          Function
Need<Function.name>                 NewSymbol
Need<Param.binding>                 NewSymbol
Need<Param.type>                    I32
Need<Params.binding>                EndParam
Need<Function.return_type>          I32
Need<Expression>                    BinaryOp
Need<BinaryOperator>                Mul
Need<Binary.left>                   SymbolRef
Need<SymbolRef.target>              @1
Need<Binary.right>                  IntLiteral
Need<IntLiteral.value>              2
Need<Expression>                    EndFunction
```

Tree-like IR model. This is what exists in the controller:

```text
FUNCTION @0
├── PARAMS
│   └── PARAM @1
│       └── TYPE I32
├── RETURN
│   └── TYPE I32
└── BLOCK
    └── BINARY_OP
        ├── OP MUL
        ├── REF @1
        └── INT 2
```

Naming pass:

```text
@0 → double
@1 → x
```

The controller can derive the action trace shown in Section 22. The renderer then recreates the source without learned punctuation decisions.

---

# 29. Architecture Summary

The preferred first baseline is decoder-only:

```text
English prompt embeddings ─────────────────────┐
                                              │
Previous structural actions ──────────────────┤
                                              ▼
Current Need and state ─────────────► Structural decoder
                                              │ action scores
                                              ▼
                                   ┌────────────────────────┐
                                   │ Controller             │
                                   │                        │
                                   │ - validity masks       │
                                   │ - custom IR updates    │
                                   │ - symbols and scope    │
                                   │ - next Need and state  │
                                   └────────────┬───────────┘
                                                │
                       ┌────────────────────────┴───────────┐
                       │                                    │
                       └──── feedback to decoder       Custom IR
                                                            │
                                               ┌────────────┴────────────┐
                                               │                         │
                                               ▼                         ▼
                                         Naming pass             Compiler analysis
                                               │                 optional annotations
                                               └────────────┬────────────┘
                                                            ▼
                                                         Renderer
                                                            │
                                                            ▼
                                                         rustfmt
                                                            │
                                                            ▼
                                                       Rust source
```

An encoder-decoder experiment can replace the prompt prefix with cross-attention to fixed prompt states. The controller feedback still enters the structural decoder.

---

# 30. Claims to Avoid Until Tested

Do not claim that this design:

- stores more literal patterns in model weights
- is more semantic than every text-based code model
- always uses fewer decoding steps
- outperforms BPE-based models
- eliminates all syntax errors
- makes type checking unnecessary
- solves arbitrary partial-code semantics
- is equivalent to rustc HIR
- is a standard mixture-of-experts architecture

A supportable current claim is:

> The architecture moves known Rust grammar and source serialization out of the neural prediction problem. The model can focus more learned decisions on program structure. Experiments must determine whether this improves efficiency or capability.
