//! Backend-independent teacher-forcing data derived from structural traces.
//!
//! This contract stops at boundaries requiring a tokenizer, numeric encoding,
//! or replay-controller observation. Bump the schema version when field
//! meanings, fixed candidate IDs/order, or target semantics change.

use std::collections::BTreeSet;

use crate::controller::{
    Action, BinaryOperatorAction, BlockAction, CallArgumentAction, DeclarationAction,
    ExpressionAction, IfElseAction, ItemListAction, Need, ParameterListAction, RootAction,
    TraceStep, TypeAction, TypeAnnotationAction,
};
use crate::model::ir::{BinaryOperator, PrimitiveType, SymbolId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SchemaVersion(pub u16);
pub const TRAINING_SCHEMA_VERSION: SchemaVersion = SchemaVersion(3);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OutputHead {
    Root,
    ItemList,
    DeclarationKind,
    ParameterList,
    Type,
    Block,
    TypeAnnotation,
    Expression,
    LiteralKind,
    BinaryOperator,
    SymbolPointer,
    DirectCallTarget,
    CallArgument,
    IfElse,
}

/// Fixed-vocabulary choice. IDs are local to each head, contiguous from zero,
/// and defined by `candidate_id`, never by Rust discriminants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedTarget {
    Root(RootAction),
    ItemList(ItemListAction),
    ParameterList(ParameterListAction),
    Type(TypeAction),
    Block(BlockAction),
    TypeAnnotation(TypeAnnotationAction),
    Expression(ExpressionAction),
    BinaryOperator(BinaryOperatorAction),
    CallArgument(CallArgumentAction),
    IfElse(IfElseAction),
}

impl FixedTarget {
    /// Number of candidates in this target's output head.
    pub fn candidate_count(self) -> u16 {
        match self {
            Self::Root(_) => 2,
            Self::ItemList(_) => 2,
            Self::ParameterList(_) => 2,
            Self::Type(_) => 11,
            Self::Block(_) => 4,
            Self::TypeAnnotation(_) => 2,
            Self::Expression(_) => 5,
            Self::BinaryOperator(_) => 12,
            Self::CallArgument(_) => 2,
            Self::IfElse(_) => 2,
        }
    }

    pub fn candidate_id(self) -> u16 {
        match self {
            Self::Root(RootAction::Module) => 0,
            Self::Root(RootAction::BlockFragment) => 1,
            Self::ItemList(ItemListAction::Function) => 0,
            Self::ItemList(ItemListAction::End) => 1,
            Self::ParameterList(ParameterListAction::Parameter) => 0,
            Self::ParameterList(ParameterListAction::End) => 1,
            Self::Type(TypeAction::Primitive(v)) => primitive_id(v),
            Self::Block(BlockAction::Let) => 0,
            Self::Block(BlockAction::ExpressionStatement) => 1,
            Self::Block(BlockAction::EndAndYield) => 2,
            Self::Block(BlockAction::EndAndDiscard) => 3,
            Self::TypeAnnotation(TypeAnnotationAction::Explicit) => 0,
            Self::TypeAnnotation(TypeAnnotationAction::Infer) => 1,
            Self::Expression(ExpressionAction::Symbol) => 0,
            Self::Expression(ExpressionAction::Literal) => 1,
            Self::Expression(ExpressionAction::Binary) => 2,
            Self::Expression(ExpressionAction::DirectCall) => 3,
            Self::Expression(ExpressionAction::If) => 4,
            Self::BinaryOperator(BinaryOperatorAction::Operator(v)) => operator_id(v),
            Self::CallArgument(CallArgumentAction::Argument) => 0,
            Self::CallArgument(CallArgumentAction::End) => 1,
            Self::IfElse(IfElseAction::Else) => 0,
            Self::IfElse(IfElseAction::NoElse) => 1,
        }
    }
}

fn primitive_id(value: PrimitiveType) -> u16 {
    match value {
        PrimitiveType::Bool => 0,
        PrimitiveType::Char => 1,
        PrimitiveType::I32 => 2,
        PrimitiveType::I64 => 3,
        PrimitiveType::Isize => 4,
        PrimitiveType::U32 => 5,
        PrimitiveType::U64 => 6,
        PrimitiveType::Usize => 7,
        PrimitiveType::F32 => 8,
        PrimitiveType::F64 => 9,
        PrimitiveType::Unit => 10,
    }
}

fn operator_id(value: BinaryOperator) -> u16 {
    match value {
        BinaryOperator::Add => 0,
        BinaryOperator::Subtract => 1,
        BinaryOperator::Multiply => 2,
        BinaryOperator::Divide => 3,
        BinaryOperator::Equal => 4,
        BinaryOperator::NotEqual => 5,
        BinaryOperator::LessThan => 6,
        BinaryOperator::LessOrEqual => 7,
        BinaryOperator::GreaterThan => 8,
        BinaryOperator::GreaterOrEqual => 9,
        BinaryOperator::And => 10,
        BinaryOperator::Or => 11,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeclarationKind {
    Function,
    Parameter,
    Local,
}
impl DeclarationKind {
    pub const CANDIDATE_COUNT: u16 = 3;
    pub fn candidate_id(self) -> u16 {
        match self {
            Self::Function => 0,
            Self::Parameter => 1,
            Self::Local => 2,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiteralKind {
    Integer,
    String,
    Bool,
}
impl LiteralKind {
    pub const CANDIDATE_COUNT: u16 = 3;
    pub fn candidate_id(self) -> u16 {
        match self {
            Self::Integer => 0,
            Self::String => 1,
            Self::Bool => 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    Fixed(FixedTarget),
    DeclarationKind(DeclarationKind),
    LiteralKind(LiteralKind),
    SymbolPointerSlot(usize),
    DirectCallTargetSlot(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Observation {
    None,
    /// Pre-action legal choices in observation order; conversion never sorts.
    DynamicCandidates(Vec<SymbolId>),
}

/// Post-action/source-only data. Model features must never consume annotations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Annotation {
    None,
    DeclaredSymbol(SymbolId),
    /// Raw value awaiting a future value-generation contract. Only its kind is
    /// supervised in this version; the payload is not claimed to be realizable.
    RawLiteral(crate::controller::LiteralAction),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TeacherForcingRecord {
    pub step_index: usize,
    pub need: Need,
    /// History is `TrainingSequence::actions[..history_end]`.
    pub history_end: usize,
    pub output_head: OutputHead,
    pub target: Target,
    /// Pre-action information that model features may consume.
    pub observation: Observation,
    /// Post-action source information that model features must never consume.
    pub annotation: Annotation,
}

/// Caller-observed legal candidates for a dynamic pointer step. A completed
/// tree cannot reconstruct every observation, notably forward functions;
/// future replay-controller observations should supply this input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedTraceStep {
    pub step: TraceStep,
    pub dynamic_candidates: Option<Vec<SymbolId>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainingSequence {
    pub schema_version: SchemaVersion,
    /// Owned once; records reference prefixes instead of cloning histories.
    pub actions: Vec<Action>,
    pub records: Vec<TeacherForcingRecord>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConversionError {
    NeedActionMismatch {
        step_index: usize,
        need: Need,
        action: Action,
    },
    UnexpectedCandidateContext {
        step_index: usize,
    },
    MissingCandidateContext {
        step_index: usize,
    },
    DuplicateCandidate {
        step_index: usize,
        symbol: SymbolId,
    },
    TargetNotInCandidates {
        step_index: usize,
        symbol: SymbolId,
    },
}

pub fn sequence_from_observed_trace(
    steps: Vec<ObservedTraceStep>,
) -> Result<TrainingSequence, ConversionError> {
    let mut actions = Vec::with_capacity(steps.len());
    let mut records = Vec::with_capacity(steps.len());
    for (step_index, observed) in steps.into_iter().enumerate() {
        let TraceStep { need, action } = observed.step;
        let (output_head, target, observation, annotation) =
            route(step_index, need, &action, observed.dynamic_candidates)?;
        records.push(TeacherForcingRecord {
            step_index,
            need,
            history_end: step_index,
            output_head,
            target,
            observation,
            annotation,
        });
        actions.push(action);
    }
    Ok(TrainingSequence {
        schema_version: TRAINING_SCHEMA_VERSION,
        actions,
        records,
    })
}

/// Convenience conversion for traces without pointer observations. It rejects
/// any pointer step because legal candidates cannot be inferred from a tree.
pub fn sequence_from_trace(steps: Vec<TraceStep>) -> Result<TrainingSequence, ConversionError> {
    sequence_from_observed_trace(
        steps
            .into_iter()
            .map(|step| ObservedTraceStep {
                step,
                dynamic_candidates: None,
            })
            .collect(),
    )
}

type Routed = (OutputHead, Target, Observation, Annotation);

fn route(
    index: usize,
    need: Need,
    action: &Action,
    candidates: Option<Vec<SymbolId>>,
) -> Result<Routed, ConversionError> {
    let mismatch = || ConversionError::NeedActionMismatch {
        step_index: index,
        need,
        action: action.clone(),
    };
    let fixed_without_candidates = |head, target| {
        reject_context(index, candidates.as_ref())?;
        Ok(fixed(head, target))
    };
    // Exhaustive outer match: adding a Need requires updating this contract.
    match need {
        Need::Root => match action {
            Action::Root(v) => fixed_without_candidates(OutputHead::Root, FixedTarget::Root(*v)),
            _ => Err(mismatch()),
        },
        Need::ItemOrEnd => match action {
            Action::ItemList(v) => {
                fixed_without_candidates(OutputHead::ItemList, FixedTarget::ItemList(*v))
            }
            _ => Err(mismatch()),
        },
        Need::Declaration => match action {
            Action::Declaration(v) => {
                reject_context(index, candidates.as_ref())?;
                let (kind, symbol) = match v {
                    DeclarationAction::Function(s) => (DeclarationKind::Function, *s),
                    DeclarationAction::Parameter(s) => (DeclarationKind::Parameter, *s),
                    DeclarationAction::Local(s) => (DeclarationKind::Local, *s),
                };
                Ok((
                    OutputHead::DeclarationKind,
                    Target::DeclarationKind(kind),
                    Observation::None,
                    Annotation::DeclaredSymbol(symbol),
                ))
            }
            _ => Err(mismatch()),
        },
        Need::ParameterOrEnd => match action {
            Action::ParameterList(v) => {
                fixed_without_candidates(OutputHead::ParameterList, FixedTarget::ParameterList(*v))
            }
            _ => Err(mismatch()),
        },
        Need::Type => match action {
            Action::Type(v) => fixed_without_candidates(OutputHead::Type, FixedTarget::Type(*v)),
            _ => Err(mismatch()),
        },
        Need::BlockEntryOrEnd => match action {
            Action::Block(v) => fixed_without_candidates(OutputHead::Block, FixedTarget::Block(*v)),
            _ => Err(mismatch()),
        },
        Need::TypeAnnotation => match action {
            Action::TypeAnnotation(v) => fixed_without_candidates(
                OutputHead::TypeAnnotation,
                FixedTarget::TypeAnnotation(*v),
            ),
            _ => Err(mismatch()),
        },
        Need::Expression => match action {
            Action::Expression(v) => {
                fixed_without_candidates(OutputHead::Expression, FixedTarget::Expression(*v))
            }
            _ => Err(mismatch()),
        },
        Need::Literal => match action {
            Action::Literal(v) => {
                reject_context(index, candidates.as_ref())?;
                let kind = match v {
                    crate::controller::LiteralAction::Integer(_) => LiteralKind::Integer,
                    crate::controller::LiteralAction::String(_) => LiteralKind::String,
                    crate::controller::LiteralAction::Bool(_) => LiteralKind::Bool,
                };
                Ok((
                    OutputHead::LiteralKind,
                    Target::LiteralKind(kind),
                    Observation::None,
                    Annotation::RawLiteral(v.clone()),
                ))
            }
            _ => Err(mismatch()),
        },
        Need::BinaryOperator => match action {
            Action::BinaryOperator(v) => fixed_without_candidates(
                OutputHead::BinaryOperator,
                FixedTarget::BinaryOperator(*v),
            ),
            _ => Err(mismatch()),
        },
        Need::SymbolReference => pointer(index, need, action, candidates, false),
        Need::DirectCallTarget => pointer(index, need, action, candidates, true),
        Need::CallArgumentOrEnd => match action {
            Action::CallArgument(v) => {
                fixed_without_candidates(OutputHead::CallArgument, FixedTarget::CallArgument(*v))
            }
            _ => Err(mismatch()),
        },
        Need::IfElse => match action {
            Action::IfElse(v) => {
                fixed_without_candidates(OutputHead::IfElse, FixedTarget::IfElse(*v))
            }
            _ => Err(mismatch()),
        },
    }
}

fn reject_context(index: usize, candidates: Option<&Vec<SymbolId>>) -> Result<(), ConversionError> {
    if candidates.is_some() {
        Err(ConversionError::UnexpectedCandidateContext { step_index: index })
    } else {
        Ok(())
    }
}

fn fixed(head: OutputHead, target: FixedTarget) -> Routed {
    (
        head,
        Target::Fixed(target),
        Observation::None,
        Annotation::None,
    )
}

fn pointer(
    index: usize,
    need: Need,
    action: &Action,
    candidates: Option<Vec<SymbolId>>,
    direct: bool,
) -> Result<Routed, ConversionError> {
    let symbol = match (direct, action) {
        (false, Action::SymbolReference(v)) | (true, Action::DirectCallTarget(v)) => v.symbol,
        _ => {
            return Err(ConversionError::NeedActionMismatch {
                step_index: index,
                need,
                action: action.clone(),
            });
        }
    };
    let candidates =
        candidates.ok_or(ConversionError::MissingCandidateContext { step_index: index })?;
    let mut seen = BTreeSet::new();
    for candidate in &candidates {
        if !seen.insert(*candidate) {
            return Err(ConversionError::DuplicateCandidate {
                step_index: index,
                symbol: *candidate,
            });
        }
    }
    let slot = candidates
        .iter()
        .position(|candidate| *candidate == symbol)
        .ok_or(ConversionError::TargetNotInCandidates {
            step_index: index,
            symbol,
        })?;
    let (head, target) = if direct {
        (
            OutputHead::DirectCallTarget,
            Target::DirectCallTargetSlot(slot),
        )
    } else {
        (OutputHead::SymbolPointer, Target::SymbolPointerSlot(slot))
    };
    Ok((
        head,
        target,
        Observation::DynamicCandidates(candidates),
        Annotation::None,
    ))
}

// Name/free-token prediction is intentionally absent. Add it only when a
// corresponding Need and tokenizer-independent target contract are specified.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::{LiteralAction, SymbolReferenceAction};
    use crate::model::ir::{Block, Expression, Function, Item, Literal, Parameter, Root};

    fn step(need: Need, action: Action) -> TraceStep {
        TraceStep { need, action }
    }
    fn observed(step: TraceStep, candidates: Option<Vec<SymbolId>>) -> ObservedTraceStep {
        ObservedTraceStep {
            step,
            dynamic_candidates: candidates,
        }
    }

    #[test]
    fn schema_envelope_and_linear_history_storage() {
        let sequence = sequence_from_trace(vec![
            step(Need::Root, Action::Root(RootAction::Module)),
            step(Need::ItemOrEnd, Action::ItemList(ItemListAction::End)),
        ])
        .unwrap();
        assert_eq!(sequence.schema_version, TRAINING_SCHEMA_VERSION);
        assert_eq!(sequence.actions.len(), 2);
        assert_eq!(
            sequence
                .records
                .iter()
                .map(|r| r.history_end)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn converts_increment_trace_with_observed_pointer_candidates() {
        let root = Root::Module {
            items: vec![Item::Function(Function {
                name: SymbolId(1),
                parameters: vec![Parameter {
                    name: SymbolId(2),
                    ty: PrimitiveType::I32,
                }],
                return_type: PrimitiveType::I32,
                body: Block {
                    statements: vec![],
                    tail: Some(Box::new(Expression::Binary {
                        left: Box::new(Expression::Symbol(SymbolId(2))),
                        operator: BinaryOperator::Add,
                        right: Box::new(Expression::Literal(Literal::Integer(1))),
                    })),
                },
            })],
        };
        let observed = crate::controller::trace(&root)
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(index, step)| observed(step, (index == 13).then(|| vec![SymbolId(2)])))
            .collect();
        let sequence = sequence_from_observed_trace(observed).unwrap();

        assert_eq!(sequence.records.len(), 16);
        assert_eq!(sequence.records[13].target, Target::SymbolPointerSlot(0));
        assert_eq!(
            sequence.records[15].annotation,
            Annotation::RawLiteral(LiteralAction::Integer(1))
        );
    }

    #[test]
    fn declaration_id_is_metadata_not_target() {
        let sequence = sequence_from_trace(vec![step(
            Need::Declaration,
            Action::Declaration(DeclarationAction::Local(SymbolId(9))),
        )])
        .unwrap();
        assert_eq!(
            sequence.records[0].target,
            Target::DeclarationKind(DeclarationKind::Local)
        );
        assert_eq!(
            sequence.records[0].annotation,
            Annotation::DeclaredSymbol(SymbolId(9))
        );
    }

    #[test]
    fn pointer_preserves_order_and_targets_slot() {
        let sequence = sequence_from_observed_trace(vec![observed(
            step(
                Need::SymbolReference,
                Action::SymbolReference(SymbolReferenceAction {
                    symbol: SymbolId(7),
                }),
            ),
            Some(vec![SymbolId(9), SymbolId(7), SymbolId(2)]),
        )])
        .unwrap();
        assert_eq!(sequence.records[0].target, Target::SymbolPointerSlot(1));
        assert_eq!(
            sequence.records[0].observation,
            Observation::DynamicCandidates(vec![SymbolId(9), SymbolId(7), SymbolId(2)])
        );
        assert_eq!(sequence.records[0].annotation, Annotation::None);
    }

    #[test]
    fn pointer_observations_cannot_be_silently_swapped() {
        let first = step(
            Need::SymbolReference,
            Action::SymbolReference(SymbolReferenceAction {
                symbol: SymbolId(1),
            }),
        );
        let second = step(
            Need::DirectCallTarget,
            Action::DirectCallTarget(SymbolReferenceAction {
                symbol: SymbolId(9),
            }),
        );
        let swapped = vec![
            observed(first, Some(vec![SymbolId(9)])),
            observed(second, Some(vec![SymbolId(1), SymbolId(2)])),
        ];
        assert_eq!(
            sequence_from_observed_trace(swapped),
            Err(ConversionError::TargetNotInCandidates {
                step_index: 0,
                symbol: SymbolId(1),
            })
        );
    }

    #[test]
    fn every_candidate_id_and_head_count_is_stable() {
        let heads = vec![
            vec![
                FixedTarget::Root(RootAction::Module),
                FixedTarget::Root(RootAction::BlockFragment),
            ],
            vec![
                FixedTarget::ItemList(ItemListAction::Function),
                FixedTarget::ItemList(ItemListAction::End),
            ],
            vec![
                FixedTarget::ParameterList(ParameterListAction::Parameter),
                FixedTarget::ParameterList(ParameterListAction::End),
            ],
            vec![
                FixedTarget::Type(TypeAction::Primitive(PrimitiveType::Bool)),
                FixedTarget::Type(TypeAction::Primitive(PrimitiveType::Char)),
                FixedTarget::Type(TypeAction::Primitive(PrimitiveType::I32)),
                FixedTarget::Type(TypeAction::Primitive(PrimitiveType::I64)),
                FixedTarget::Type(TypeAction::Primitive(PrimitiveType::Isize)),
                FixedTarget::Type(TypeAction::Primitive(PrimitiveType::U32)),
                FixedTarget::Type(TypeAction::Primitive(PrimitiveType::U64)),
                FixedTarget::Type(TypeAction::Primitive(PrimitiveType::Usize)),
                FixedTarget::Type(TypeAction::Primitive(PrimitiveType::F32)),
                FixedTarget::Type(TypeAction::Primitive(PrimitiveType::F64)),
                FixedTarget::Type(TypeAction::Primitive(PrimitiveType::Unit)),
            ],
            vec![
                FixedTarget::Block(BlockAction::Let),
                FixedTarget::Block(BlockAction::ExpressionStatement),
                FixedTarget::Block(BlockAction::EndAndYield),
                FixedTarget::Block(BlockAction::EndAndDiscard),
            ],
            vec![
                FixedTarget::TypeAnnotation(TypeAnnotationAction::Explicit),
                FixedTarget::TypeAnnotation(TypeAnnotationAction::Infer),
            ],
            vec![
                FixedTarget::Expression(ExpressionAction::Symbol),
                FixedTarget::Expression(ExpressionAction::Literal),
                FixedTarget::Expression(ExpressionAction::Binary),
                FixedTarget::Expression(ExpressionAction::DirectCall),
                FixedTarget::Expression(ExpressionAction::If),
            ],
            vec![
                FixedTarget::BinaryOperator(BinaryOperatorAction::Operator(BinaryOperator::Add)),
                FixedTarget::BinaryOperator(BinaryOperatorAction::Operator(
                    BinaryOperator::Subtract,
                )),
                FixedTarget::BinaryOperator(BinaryOperatorAction::Operator(
                    BinaryOperator::Multiply,
                )),
                FixedTarget::BinaryOperator(BinaryOperatorAction::Operator(BinaryOperator::Divide)),
                FixedTarget::BinaryOperator(BinaryOperatorAction::Operator(BinaryOperator::Equal)),
                FixedTarget::BinaryOperator(BinaryOperatorAction::Operator(
                    BinaryOperator::NotEqual,
                )),
                FixedTarget::BinaryOperator(BinaryOperatorAction::Operator(
                    BinaryOperator::LessThan,
                )),
                FixedTarget::BinaryOperator(BinaryOperatorAction::Operator(
                    BinaryOperator::LessOrEqual,
                )),
                FixedTarget::BinaryOperator(BinaryOperatorAction::Operator(
                    BinaryOperator::GreaterThan,
                )),
                FixedTarget::BinaryOperator(BinaryOperatorAction::Operator(
                    BinaryOperator::GreaterOrEqual,
                )),
                FixedTarget::BinaryOperator(BinaryOperatorAction::Operator(BinaryOperator::And)),
                FixedTarget::BinaryOperator(BinaryOperatorAction::Operator(BinaryOperator::Or)),
            ],
            vec![
                FixedTarget::CallArgument(CallArgumentAction::Argument),
                FixedTarget::CallArgument(CallArgumentAction::End),
            ],
            vec![
                FixedTarget::IfElse(IfElseAction::Else),
                FixedTarget::IfElse(IfElseAction::NoElse),
            ],
        ];
        for head in heads {
            let expected: Vec<u16> = (0..head.len() as u16).collect();
            assert_eq!(
                head.iter()
                    .map(|target| target.candidate_id())
                    .collect::<Vec<_>>(),
                expected
            );
            assert!(
                head.iter()
                    .all(|target| target.candidate_count() == head.len() as u16)
            );
        }

        let declarations = [
            DeclarationKind::Function,
            DeclarationKind::Parameter,
            DeclarationKind::Local,
        ];
        assert_eq!(DeclarationKind::CANDIDATE_COUNT, declarations.len() as u16);
        assert_eq!(declarations.map(DeclarationKind::candidate_id), [0, 1, 2]);
        let literals = [LiteralKind::Integer, LiteralKind::String, LiteralKind::Bool];
        assert_eq!(LiteralKind::CANDIDATE_COUNT, literals.len() as u16);
        assert_eq!(literals.map(LiteralKind::candidate_id), [0, 1, 2]);
    }

    #[test]
    fn rejects_missing_illegal_and_unexpected_context() {
        let pointer = || {
            step(
                Need::DirectCallTarget,
                Action::DirectCallTarget(SymbolReferenceAction {
                    symbol: SymbolId(4),
                }),
            )
        };
        assert_eq!(
            sequence_from_trace(vec![pointer()]),
            Err(ConversionError::MissingCandidateContext { step_index: 0 })
        );
        assert_eq!(
            sequence_from_observed_trace(vec![observed(pointer(), Some(vec![SymbolId(3)]))]),
            Err(ConversionError::TargetNotInCandidates {
                step_index: 0,
                symbol: SymbolId(4)
            })
        );
        assert_eq!(
            sequence_from_observed_trace(vec![observed(
                step(Need::Root, Action::Root(RootAction::Module)),
                Some(vec![])
            )]),
            Err(ConversionError::UnexpectedCandidateContext { step_index: 0 })
        );
    }

    #[test]
    fn literal_payload_is_pending_metadata() {
        let value = LiteralAction::String("un-tokenized".into());
        let sequence =
            sequence_from_trace(vec![step(Need::Literal, Action::Literal(value.clone()))]).unwrap();
        assert_eq!(
            sequence.records[0].target,
            Target::LiteralKind(LiteralKind::String)
        );
        assert_eq!(
            sequence.records[0].annotation,
            Annotation::RawLiteral(value)
        );
        assert_eq!(sequence.records[0].observation, Observation::None);
    }

    #[test]
    fn positive_and_mismatch_fixtures() {
        let sequence = sequence_from_trace(vec![step(
            Need::Expression,
            Action::Expression(ExpressionAction::If),
        )])
        .unwrap();
        assert_eq!(
            sequence.records[0].target,
            Target::Fixed(FixedTarget::Expression(ExpressionAction::If))
        );
        let action = Action::Root(RootAction::Module);
        assert_eq!(
            sequence_from_trace(vec![step(Need::Literal, action.clone())]),
            Err(ConversionError::NeedActionMismatch {
                step_index: 0,
                need: Need::Literal,
                action
            })
        );
    }
}
