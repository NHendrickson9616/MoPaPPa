# Architecture Plan: Pretrained English Backbone and Routed Rust Action Heads

## Purpose

Use pretrained language-model weights for English understanding while training a new, explicitly structured interface for Rust semantic generation. Rust IR actions remain distinct from free-text tokens.

This plan complements the project specification. The custom IR, deterministic controller, renderer, symbol identity model, and grammar masking remain unchanged.

## Model Components

### Possible BPE
https://crates.io/crates/bpe

### Reused pretrained components

Initialize these from a compatible pretrained English model:

- English tokenizer and English token embedding table
- Transformer attention and feed-forward blocks
- free-token output head, where the model must generate ordinary text

### New trainable Rust-semantic components

Initialize these components randomly:

- Rust structural-action embeddings
- `Need` embeddings
- controller-state embeddings, such as parent node, current field, scope, and expected category
- category-specific Rust action heads
- symbol-pointer head
- any literal, naming, or list-continuation heads that are not free-token generation

All input representations must have the pretrained transformer's `d_model` width. Input modality remains explicit:

```text
English token       → pretrained English embedding + English modality embedding
Rust action         → learned action embedding + Rust-action modality embedding
Need/state feature  → learned state embedding
```

The structural decoder then consumes the English condition, prior structural actions, the current `Need`, and controller state.

## Output-Head Routing

The controller determines the structural category required at every decoding step. It returns a typed `Need`; the model uses it to route the transformer's hidden state to exactly the relevant output head.

```text
English condition + previous actions + Need + state
                         │
                         ▼
                pretrained Transformer body
                         │
                         ▼
                   hidden state h
                         │
                         ▼
              controller-selected output head
                         │
                         ▼
               logits for valid action category
                         │
                         ▼
           controller context mask and action application
```

Examples:

```text
Need<Item>             → ItemHead
Need<Expression>       → ExpressionHead
Need<Type>             → TypeHead
Need<BinaryOperator>   → BinaryOperatorHead
Need<SymbolReference>  → SymbolPointerHead over visible symbols
Need<FreeToken>        → pretrained FreeTokenHead
Need<ListContinuation> → ListContinuationHead
```

The controller chooses the output *space*, not the action inside it. The selected learned head chooses the most appropriate candidate in that space. The controller may further mask candidates that are invalid in the current scope or syntactic position before softmax.

For example, `Need<BinaryOperator>` invokes only the binary-operator head. The model chooses between `Add`, `Sub`, `Mul`, `Div`, and other allowed operators; it does not score English tokens, item kinds, types, or symbol references.

## Why Separate Heads

Separate action heads should be preferred over placing all Rust actions into the pretrained text vocabulary.

- Structural actions, symbol references, literals, and free text have different candidate sets and constraints.
- The controller can supply category- and context-specific masks naturally.
- The large pretrained free-token vocabulary projection is evaluated only when free text is actually needed.
- Output-projection work and memory traffic are reduced on structural decoding steps.
- Adding or reorganizing Rust actions does not alter the English tokenizer or require assigning structural semantics to text token IDs.

Skipping the free-token head can be a meaningful saving because it can contain tens of thousands of tokens. Routing among small Rust heads is primarily a correctness and modeling-clarity benefit; Transformer layers will generally remain the dominant per-step computation.

## Multi-Mechanism Decisions

Some high-level requirements may permit alternatives from different mechanisms. For example, an expression may be a structural node, a literal, or a reference to a visible symbol.

Start with a hierarchical action sequence:

```text
Need<ExpressionForm>
    → StructuralNode | Literal | SymbolReference

StructuralNode  → Need<ExpressionNodeKind>
Literal         → Need<LiteralKind>
SymbolReference → Need<SymbolReference>
```

This keeps every choice in one well-defined head, makes masking straightforward, and avoids needing scores from independently trained heads to be calibrated against each other. A combined candidate distribution across heads can be evaluated later if reducing decoding steps is shown to matter.

## Training Plan

1. Load the pretrained English tokenizer, embeddings, Transformer body, and free-token head.
2. Add the new Rust semantic embeddings and routed output heads.
3. Freeze pretrained parameters initially.
4. Train the new Rust components on paired English-to-custom-IR examples.
5. Gradually unfreeze upper Transformer layers once the action interface has begun to learn.
6. Fine-tune the complete model with separate optimizer parameter groups and a lower learning rate for pretrained parameters.

Illustrative learning-rate relationship:

```text
new Rust components:       higher learning rate, e.g. 1e-3
pretrained model weights:  lower learning rate, e.g. 1e-5 to 1e-4
```

These rates are starting points to tune experimentally, not fixed requirements.

Freezing protects pretrained English representations while the randomly initialized Rust interface is learning. Eventual unfreezing lets English representations adapt to the project's custom IR and action sequence.

## Important Limitation

English pretraining supplies useful language representations and possibly general reasoning features. It does not supply the mapping from English requests to this project's custom Rust IR.

Paired training data must teach:

- the meanings of Rust structural actions
- the English-to-IR mapping
- controller-state interpretation
- structural action sequencing
- symbol-pointer behavior
- action choices required for idiomatic Rust

## Initial Experiment

The preferred first transfer-learning experiment is:

- a pretrained decoder-only English Transformer used as the shared backbone
- pretrained English token embeddings and free-token output head
- learned, separate Rust action and controller-state embeddings
- controller-selected, category-specific output heads
- a symbol-pointer head for internal symbol identities
- controller masking after the selected head produces logits
- a frozen-backbone phase followed by gradual unfreezing

Compare this against the existing prompt-prefix baseline and, later, against a separate pretrained English encoder with a new structural decoder.
