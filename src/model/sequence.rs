//! Backend-independent causal sequencing; intentionally not exported yet.
//! `OutputHead` is the shared routing authority for sequencing and decoder
//! structural projections.

use crate::controller::{Action, DeclarationAction, LiteralAction, Need};
use crate::model::bridge::{CandidateSet, ModelRoute, ModelStep};
use crate::training::{
    Annotation, DeclarationKind, FixedTarget, LiteralKind, Observation, OutputHead, SchemaVersion,
    TRAINING_SCHEMA_VERSION, Target, TrainingSequence,
};
use std::collections::BTreeSet;

pub type TokenId = u32;
pub const SEQUENCE_PROTOCOL_VERSION: u16 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenizerIdentity {
    identifier: String,
    revision: String,
    vocabulary_hash_or_version: String,
}
impl TokenizerIdentity {
    pub fn new(
        identifier: impl Into<String>,
        revision: impl Into<String>,
        vocabulary: impl Into<String>,
    ) -> Result<Self, SequenceError> {
        let v = Self {
            identifier: identifier.into(),
            revision: revision.into(),
            vocabulary_hash_or_version: vocabulary.into(),
        };
        if v.identifier.is_empty()
            || v.revision.is_empty()
            || v.vocabulary_hash_or_version.is_empty()
        {
            Err(SequenceError::InvalidTokenizerIdentity)
        } else {
            Ok(v)
        }
    }
    pub fn identifier(&self) -> &str {
        &self.identifier
    }
    pub fn revision(&self) -> &str {
        &self.revision
    }
    pub fn vocabulary_hash_or_version(&self) -> &str {
        &self.vocabulary_hash_or_version
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodingMetadata {
    protocol_version: u16,
    training_schema_version: SchemaVersion,
    tokenizer: TokenizerIdentity,
}
impl EncodingMetadata {
    pub fn protocol_version(&self) -> u16 {
        self.protocol_version
    }
    pub fn training_schema_version(&self) -> SchemaVersion {
        self.training_schema_version
    }
    pub fn tokenizer(&self) -> &TokenizerIdentity {
        &self.tokenizer
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SupportedMetadata {
    pub protocol_version: u16,
    pub training_schema_version: SchemaVersion,
    pub tokenizer: TokenizerIdentity,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreviousAction {
    Bos,
    Action(Action),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControllerStateFeatures {
    pub completed_actions: u32,
    pub dynamic_candidate_count: u32,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuralPosition {
    previous_action: PreviousAction,
    need: Need,
    output_head: OutputHead,
    state: ControllerStateFeatures,
    observation: Observation,
}
impl StructuralPosition {
    pub fn previous_action(&self) -> &PreviousAction {
        &self.previous_action
    }
    pub fn need(&self) -> Need {
        self.need
    }
    pub fn output_head(&self) -> OutputHead {
        self.output_head
    }
    pub fn state(&self) -> ControllerStateFeatures {
        self.state
    }
    pub fn observation(&self) -> &Observation {
        &self.observation
    }

    pub(crate) fn from_model_step(
        step: &ModelStep,
        previous_action: PreviousAction,
        completed_actions: usize,
    ) -> Result<Self, SequenceError> {
        let ModelStep::Needs {
            need,
            route,
            candidates,
            ..
        } = step
        else {
            return Err(SequenceError::CompleteInferenceStep);
        };
        let (output_head, observation, dynamic_candidate_count) = match (need, route, candidates) {
            (_, ModelRoute::Fixed(output_head), CandidateSet::Fixed(_)) => {
                (*output_head, Observation::None, 0)
            }
            (Need::SymbolReference, ModelRoute::DynamicPointer, CandidateSet::Symbols(symbols)) => {
                (
                    OutputHead::SymbolPointer,
                    Observation::DynamicCandidates(symbols.clone()),
                    symbols.len(),
                )
            }
            (
                Need::DirectCallTarget,
                ModelRoute::DynamicPointer,
                CandidateSet::Symbols(symbols),
            ) => (
                OutputHead::DirectCallTarget,
                Observation::DynamicCandidates(symbols.clone()),
                symbols.len(),
            ),
            (
                Need::Literal,
                ModelRoute::Value(super::bridge::ValueRequest::Integer),
                CandidateSet::Value(super::bridge::ValueRequest::Integer),
            ) => (OutputHead::LiteralKind, Observation::None, 0),
            _ => return Err(SequenceError::UnsupportedInferenceRoute),
        };
        Ok(Self {
            previous_action,
            need: *need,
            output_head,
            state: ControllerStateFeatures {
                completed_actions: u32::try_from(completed_actions)
                    .map_err(|_| SequenceError::LengthOverflow)?,
                dynamic_candidate_count: u32::try_from(dynamic_candidate_count)
                    .map_err(|_| SequenceError::LengthOverflow)?,
            },
            observation,
        })
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Position {
    EnglishToken(TokenId),
    Structural(StructuralPosition),
    SymbolNameToken(TokenId),
    FreeToken(TokenId),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuralSupervision {
    target: Target,
    action: Action,
}
impl StructuralSupervision {
    pub fn target(&self) -> &Target {
        &self.target
    }
    pub fn action(&self) -> &Action {
        &self.action
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequencePosition {
    input: Position,
    supervision: Option<StructuralSupervision>,
    /// Source-only metadata is structurally outside both model input and
    /// prediction supervision.
    source_annotation: Annotation,
}
impl SequencePosition {
    pub fn input(&self) -> &Position {
        &self.input
    }
    pub fn supervision(&self) -> Option<&StructuralSupervision> {
        self.supervision.as_ref()
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CausalSequence {
    metadata: EncodingMetadata,
    positions: Vec<SequencePosition>,
    inference_actions: Option<Vec<Action>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SequenceError {
    InvalidTokenizerIdentity,
    UnsupportedProtocolVersion,
    UnsupportedTrainingSchema,
    UnsupportedTokenizer,
    LengthOverflow,
    RecordCountMismatch,
    StepIndexMismatch { record: usize },
    HistoryAlignment { record: usize },
    MalformedRoute { record: usize },
    DuplicateCandidate { record: usize },
    UnsupportedRawLiteral { record: usize },
    UnsupportedSymbolNaming { record: usize },
    EnglishAfterStructural { position: usize },
    MissingStructuralSupervision { position: usize },
    SupervisionOnNonStructural { position: usize },
    ReservedPositionInV1 { position: usize },
    PreviousActionMismatch { position: usize },
    StateHistoryMismatch { position: usize },
    ObservationCountMismatch { position: usize },
    MetadataMismatch { sequence: usize },
    CompleteInferenceStep,
    UnsupportedInferenceRoute,
}

impl CausalSequence {
    pub fn metadata(&self) -> &EncodingMetadata {
        &self.metadata
    }
    pub fn positions(&self) -> &[SequencePosition] {
        &self.positions
    }
    pub fn from_training(
        prefix: Vec<TokenId>,
        tokenizer: TokenizerIdentity,
        training: &TrainingSequence,
    ) -> Result<Self, SequenceError> {
        if training.records.len() != training.actions.len() {
            return Err(SequenceError::RecordCountMismatch);
        }
        let mut positions: Vec<_> = prefix
            .into_iter()
            .map(|t| SequencePosition {
                input: Position::EnglishToken(t),
                supervision: None,
                source_annotation: Annotation::None,
            })
            .collect();
        for (i, (r, action)) in training.records.iter().zip(&training.actions).enumerate() {
            if r.step_index != i {
                return Err(SequenceError::StepIndexMismatch { record: i });
            }
            if r.history_end != i {
                return Err(SequenceError::HistoryAlignment { record: i });
            }
            validate_route(
                i,
                r.need,
                r.output_head,
                &r.target,
                &r.observation,
                &r.annotation,
                action,
            )?;
            match r.annotation {
                Annotation::RawLiteral(_) => {
                    return Err(SequenceError::UnsupportedRawLiteral { record: i });
                }
                Annotation::DeclaredSymbol(_) => {
                    return Err(SequenceError::UnsupportedSymbolNaming { record: i });
                }
                Annotation::None => {}
            }
            let count = observation_count(&r.observation);
            positions.push(SequencePosition {
                input: Position::Structural(StructuralPosition {
                    previous_action: if i == 0 {
                        PreviousAction::Bos
                    } else {
                        PreviousAction::Action(training.actions[i - 1].clone())
                    },
                    need: r.need,
                    output_head: r.output_head,
                    state: ControllerStateFeatures {
                        completed_actions: u32::try_from(i)
                            .map_err(|_| SequenceError::LengthOverflow)?,
                        dynamic_candidate_count: u32::try_from(count)
                            .map_err(|_| SequenceError::LengthOverflow)?,
                    },
                    observation: r.observation.clone(),
                }),
                supervision: Some(StructuralSupervision {
                    target: r.target.clone(),
                    action: action.clone(),
                }),
                source_annotation: r.annotation.clone(),
            });
        }
        let value = Self {
            metadata: EncodingMetadata {
                protocol_version: SEQUENCE_PROTOCOL_VERSION,
                training_schema_version: training.schema_version,
                tokenizer,
            },
            positions,
            inference_actions: None,
        };
        value.validate()?;
        Ok(value)
    }

    /// Builds an unsupervised causal input ending at the current live decision.
    ///
    /// Unlike teacher-forcing construction, structural positions deliberately
    /// carry no target. `history` contains the exact inputs previously scored
    /// alongside the actions that were accepted from them.
    pub(crate) fn from_inference(
        prefix: &[TokenId],
        tokenizer: &TokenizerIdentity,
        history: &[(StructuralPosition, Action)],
        current: StructuralPosition,
    ) -> Self {
        let mut positions: Vec<_> = prefix
            .iter()
            .copied()
            .map(|token| SequencePosition {
                input: Position::EnglishToken(token),
                supervision: None,
                source_annotation: Annotation::None,
            })
            .collect();
        positions.extend(history.iter().map(|(input, _)| SequencePosition {
            input: Position::Structural(input.clone()),
            supervision: None,
            source_annotation: Annotation::None,
        }));
        positions.push(SequencePosition {
            input: Position::Structural(current),
            supervision: None,
            source_annotation: Annotation::None,
        });
        Self {
            metadata: EncodingMetadata {
                protocol_version: SEQUENCE_PROTOCOL_VERSION,
                training_schema_version: TRAINING_SCHEMA_VERSION,
                tokenizer: tokenizer.clone(),
            },
            positions,
            inference_actions: Some(history.iter().map(|(_, action)| action.clone()).collect()),
        }
    }
    pub fn validate(&self) -> Result<(), SequenceError> {
        let mut prior: Option<&Action> = None;
        let mut n = 0;
        for (p, item) in self.positions.iter().enumerate() {
            match &item.input {
                Position::EnglishToken(_) => {
                    if n != 0 {
                        return Err(SequenceError::EnglishAfterStructural { position: p });
                    }
                    if item.supervision.is_some() {
                        return Err(SequenceError::SupervisionOnNonStructural { position: p });
                    }
                }
                Position::SymbolNameToken(_) | Position::FreeToken(_) => {
                    if self.metadata.protocol_version == SEQUENCE_PROTOCOL_VERSION {
                        return Err(SequenceError::ReservedPositionInV1 { position: p });
                    }
                    if item.supervision.is_some() {
                        return Err(SequenceError::SupervisionOnNonStructural { position: p });
                    }
                }
                Position::Structural(input) => {
                    let expected =
                        prior.map_or(PreviousAction::Bos, |a| PreviousAction::Action(a.clone()));
                    if input.previous_action != expected {
                        return Err(SequenceError::PreviousActionMismatch { position: p });
                    }
                    if input.state.completed_actions as usize != n {
                        return Err(SequenceError::StateHistoryMismatch { position: p });
                    }
                    if input.state.dynamic_candidate_count as usize
                        != observation_count(&input.observation)
                    {
                        return Err(SequenceError::ObservationCountMismatch { position: p });
                    }
                    if let Some(actions) = &self.inference_actions {
                        if item.supervision.is_some() {
                            return Err(SequenceError::SupervisionOnNonStructural { position: p });
                        }
                        prior = actions.get(n);
                        n += 1;
                        continue;
                    }
                    let s = item
                        .supervision
                        .as_ref()
                        .ok_or(SequenceError::MissingStructuralSupervision { position: p })?;
                    validate_route(
                        n,
                        input.need,
                        input.output_head,
                        &s.target,
                        &input.observation,
                        &item.source_annotation,
                        &s.action,
                    )?;
                    prior = Some(&s.action);
                    n += 1;
                }
            }
        }
        Ok(())
    }
    pub fn validate_supported(&self, s: &SupportedMetadata) -> Result<(), SequenceError> {
        self.validate()?;
        if self.metadata.protocol_version != s.protocol_version {
            return Err(SequenceError::UnsupportedProtocolVersion);
        }
        if self.metadata.training_schema_version != s.training_schema_version {
            return Err(SequenceError::UnsupportedTrainingSchema);
        }
        if self.metadata.tokenizer != s.tokenizer {
            return Err(SequenceError::UnsupportedTokenizer);
        }
        Ok(())
    }
}
pub fn validate_batch(values: &[CausalSequence]) -> Result<(), SequenceError> {
    if let Some(first) = values.first() {
        for (i, v) in values.iter().enumerate() {
            if v.metadata != first.metadata {
                return Err(SequenceError::MetadataMismatch { sequence: i });
            }
            v.validate()?;
        }
    }
    Ok(())
}
fn observation_count(o: &Observation) -> usize {
    match o {
        Observation::None => 0,
        Observation::DynamicCandidates(v) => v.len(),
    }
}
fn validate_route(
    i: usize,
    need: Need,
    head: OutputHead,
    target: &Target,
    obs: &Observation,
    ann: &Annotation,
    action: &Action,
) -> Result<(), SequenceError> {
    if head != expected_head(need) {
        return Err(SequenceError::MalformedRoute { record: i });
    }
    let valid = match (need, action, target, ann) {
        (Need::Root, Action::Root(a), Target::Fixed(FixedTarget::Root(t)), Annotation::None) => {
            a == t
        }
        (
            Need::ItemOrEnd,
            Action::ItemList(a),
            Target::Fixed(FixedTarget::ItemList(t)),
            Annotation::None,
        ) => a == t,
        (
            Need::Declaration,
            Action::Declaration(a),
            Target::DeclarationKind(t),
            Annotation::DeclaredSymbol(s),
        ) => declaration(a) == (*t, *s),
        (
            Need::ParameterOrEnd,
            Action::ParameterList(a),
            Target::Fixed(FixedTarget::ParameterList(t)),
            Annotation::None,
        ) => a == t,
        (Need::Type, Action::Type(a), Target::Fixed(FixedTarget::Type(t)), Annotation::None) => {
            a == t
        }
        (
            Need::BlockEntryOrEnd,
            Action::Block(a),
            Target::Fixed(FixedTarget::Block(t)),
            Annotation::None,
        ) => a == t,
        (
            Need::TypeAnnotation,
            Action::TypeAnnotation(a),
            Target::Fixed(FixedTarget::TypeAnnotation(t)),
            Annotation::None,
        ) => a == t,
        (
            Need::Expression,
            Action::Expression(a),
            Target::Fixed(FixedTarget::Expression(t)),
            Annotation::None,
        ) => a == t,
        (
            Need::Literal,
            Action::Literal(a),
            Target::LiteralKind(t),
            Annotation::RawLiteral(raw),
        ) => a == raw && literal(a) == *t,
        (
            Need::BinaryOperator,
            Action::BinaryOperator(a),
            Target::Fixed(FixedTarget::BinaryOperator(t)),
            Annotation::None,
        ) => a == t,
        (
            Need::CallArgumentOrEnd,
            Action::CallArgument(a),
            Target::Fixed(FixedTarget::CallArgument(t)),
            Annotation::None,
        ) => a == t,
        (
            Need::IfElse,
            Action::IfElse(a),
            Target::Fixed(FixedTarget::IfElse(t)),
            Annotation::None,
        ) => a == t,
        (
            Need::SymbolReference,
            Action::SymbolReference(a),
            Target::SymbolPointerSlot(slot),
            Annotation::None,
        )
        | (
            Need::DirectCallTarget,
            Action::DirectCallTarget(a),
            Target::DirectCallTargetSlot(slot),
            Annotation::None,
        ) => pointer(i, a.symbol, *slot, obs)?,
        _ => false,
    };
    let pointer_head = matches!(
        head,
        OutputHead::SymbolPointer | OutputHead::DirectCallTarget
    );
    let obs_ok = matches!(
        (pointer_head, obs),
        (true, Observation::DynamicCandidates(_)) | (false, Observation::None)
    );
    if valid && obs_ok && target_head(head, target) {
        Ok(())
    } else {
        Err(SequenceError::MalformedRoute { record: i })
    }
}
fn pointer(
    i: usize,
    symbol: crate::model::ir::SymbolId,
    slot: usize,
    obs: &Observation,
) -> Result<bool, SequenceError> {
    let Observation::DynamicCandidates(v) = obs else {
        return Ok(false);
    };
    let mut seen = BTreeSet::new();
    if v.iter().any(|x| !seen.insert(*x)) {
        return Err(SequenceError::DuplicateCandidate { record: i });
    }
    Ok(v.get(slot) == Some(&symbol))
}
fn target_head(h: OutputHead, t: &Target) -> bool {
    matches!(
        (h, t),
        (OutputHead::Root, Target::Fixed(FixedTarget::Root(_)))
            | (
                OutputHead::ItemList,
                Target::Fixed(FixedTarget::ItemList(_))
            )
            | (
                OutputHead::ParameterList,
                Target::Fixed(FixedTarget::ParameterList(_))
            )
            | (OutputHead::Type, Target::Fixed(FixedTarget::Type(_)))
            | (OutputHead::Block, Target::Fixed(FixedTarget::Block(_)))
            | (
                OutputHead::TypeAnnotation,
                Target::Fixed(FixedTarget::TypeAnnotation(_))
            )
            | (
                OutputHead::Expression,
                Target::Fixed(FixedTarget::Expression(_))
            )
            | (
                OutputHead::BinaryOperator,
                Target::Fixed(FixedTarget::BinaryOperator(_))
            )
            | (
                OutputHead::CallArgument,
                Target::Fixed(FixedTarget::CallArgument(_))
            )
            | (OutputHead::IfElse, Target::Fixed(FixedTarget::IfElse(_)))
            | (OutputHead::DeclarationKind, Target::DeclarationKind(_))
            | (OutputHead::LiteralKind, Target::LiteralKind(_))
            | (OutputHead::SymbolPointer, Target::SymbolPointerSlot(_))
            | (
                OutputHead::DirectCallTarget,
                Target::DirectCallTargetSlot(_)
            )
    )
}
fn declaration(a: &DeclarationAction) -> (DeclarationKind, crate::model::ir::SymbolId) {
    match a {
        DeclarationAction::Function(s) => (DeclarationKind::Function, *s),
        DeclarationAction::Parameter(s) => (DeclarationKind::Parameter, *s),
        DeclarationAction::Local(s) => (DeclarationKind::Local, *s),
    }
}
fn literal(a: &LiteralAction) -> LiteralKind {
    match a {
        LiteralAction::Integer(_) => LiteralKind::Integer,
        LiteralAction::String(_) => LiteralKind::String,
        LiteralAction::Bool(_) => LiteralKind::Bool,
    }
}
fn expected_head(n: Need) -> OutputHead {
    match n {
        Need::Root => OutputHead::Root,
        Need::ItemOrEnd => OutputHead::ItemList,
        Need::Declaration => OutputHead::DeclarationKind,
        Need::ParameterOrEnd => OutputHead::ParameterList,
        Need::Type => OutputHead::Type,
        Need::BlockEntryOrEnd => OutputHead::Block,
        Need::TypeAnnotation => OutputHead::TypeAnnotation,
        Need::Expression => OutputHead::Expression,
        Need::Literal => OutputHead::LiteralKind,
        Need::BinaryOperator => OutputHead::BinaryOperator,
        Need::SymbolReference => OutputHead::SymbolPointer,
        Need::DirectCallTarget => OutputHead::DirectCallTarget,
        Need::CallArgumentOrEnd => OutputHead::CallArgument,
        Need::IfElse => OutputHead::IfElse,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::{ItemListAction, RootAction, SymbolReferenceAction};
    use crate::model::ir::SymbolId;
    use crate::training::{TRAINING_SCHEMA_VERSION, TeacherForcingRecord};

    fn tokenizer() -> TokenizerIdentity {
        TokenizerIdentity::new("cl100k_base", "2024-01", "sha256:abc").unwrap()
    }

    fn basic() -> TrainingSequence {
        TrainingSequence {
            schema_version: TRAINING_SCHEMA_VERSION,
            actions: vec![
                Action::Root(RootAction::Module),
                Action::ItemList(ItemListAction::End),
            ],
            records: vec![
                TeacherForcingRecord {
                    step_index: 0,
                    need: Need::Root,
                    history_end: 0,
                    output_head: OutputHead::Root,
                    target: Target::Fixed(FixedTarget::Root(RootAction::Module)),
                    observation: Observation::None,
                    annotation: Annotation::None,
                },
                TeacherForcingRecord {
                    step_index: 1,
                    need: Need::ItemOrEnd,
                    history_end: 1,
                    output_head: OutputHead::ItemList,
                    target: Target::Fixed(FixedTarget::ItemList(ItemListAction::End)),
                    observation: Observation::None,
                    annotation: Annotation::None,
                },
            ],
        }
    }

    fn convert(training: &TrainingSequence) -> Result<CausalSequence, SequenceError> {
        CausalSequence::from_training(vec![10, 11], tokenizer(), training)
    }

    #[test]
    fn prefix_bos_history_no_leakage_and_linear_storage() {
        let training = basic();
        let sequence = convert(&training).unwrap();
        assert_eq!(sequence.positions().len(), 2 + training.records.len());
        assert!(matches!(
            sequence.positions()[0].input(),
            Position::EnglishToken(10)
        ));
        let Position::Structural(first) = sequence.positions()[2].input() else {
            panic!("structural position")
        };
        let Position::Structural(second) = sequence.positions()[3].input() else {
            panic!("structural position")
        };
        assert_eq!(first.previous_action(), &PreviousAction::Bos);
        assert_eq!(
            second.previous_action(),
            &PreviousAction::Action(training.actions[0].clone())
        );
        assert_ne!(
            second.previous_action(),
            &PreviousAction::Action(training.actions[1].clone())
        );
        assert_eq!(first.state().completed_actions, 0);
        assert_eq!(second.state().completed_actions, 1);
    }

    #[test]
    fn malformed_source_payload_cannot_enter_next_previous_action() {
        let literal = LiteralAction::String("secret".into());
        let mut training = basic();
        training.actions[0] = Action::Literal(literal.clone());
        training.records[0].need = Need::Literal;
        training.records[0].output_head = OutputHead::LiteralKind;
        training.records[0].target = Target::LiteralKind(LiteralKind::Bool);
        training.records[0].annotation = Annotation::RawLiteral(literal);
        assert_eq!(
            convert(&training),
            Err(SequenceError::MalformedRoute { record: 0 })
        );

        let mut training = basic();
        training.actions[0] = Action::Declaration(DeclarationAction::Local(SymbolId(3)));
        training.records[0].need = Need::Declaration;
        training.records[0].output_head = OutputHead::DeclarationKind;
        training.records[0].target = Target::DeclarationKind(DeclarationKind::Local);
        training.records[0].annotation = Annotation::DeclaredSymbol(SymbolId(4));
        assert_eq!(
            convert(&training),
            Err(SequenceError::MalformedRoute { record: 0 })
        );
    }

    #[test]
    fn exact_previous_action_and_route_are_validated() {
        let mut sequence = convert(&basic()).unwrap();
        let Position::Structural(second) = &mut sequence.positions[3].input else {
            panic!("structural position")
        };
        second.previous_action = PreviousAction::Action(Action::Root(RootAction::BlockFragment));
        assert_eq!(
            sequence.validate(),
            Err(SequenceError::PreviousActionMismatch { position: 3 })
        );

        let mut training = basic();
        training.records[0].output_head = OutputHead::Expression;
        assert_eq!(
            convert(&training),
            Err(SequenceError::MalformedRoute { record: 0 })
        );
        let mut training = basic();
        training.records[0].target = Target::Fixed(FixedTarget::ItemList(ItemListAction::End));
        assert_eq!(
            convert(&training),
            Err(SequenceError::MalformedRoute { record: 0 })
        );
        let mut training = basic();
        training.records[1].history_end = 0;
        assert_eq!(
            convert(&training),
            Err(SequenceError::HistoryAlignment { record: 1 })
        );
    }

    fn pointer(candidates: Observation, slot: usize, symbol: SymbolId) -> TrainingSequence {
        TrainingSequence {
            schema_version: TRAINING_SCHEMA_VERSION,
            actions: vec![Action::SymbolReference(SymbolReferenceAction { symbol })],
            records: vec![TeacherForcingRecord {
                step_index: 0,
                need: Need::SymbolReference,
                history_end: 0,
                output_head: OutputHead::SymbolPointer,
                target: Target::SymbolPointerSlot(slot),
                observation: candidates,
                annotation: Annotation::None,
            }],
        }
    }

    #[test]
    fn pointer_observations_are_exclusive_unique_and_aligned() {
        let valid = pointer(
            Observation::DynamicCandidates(vec![SymbolId(1), SymbolId(2)]),
            1,
            SymbolId(2),
        );
        CausalSequence::from_training(vec![], tokenizer(), &valid).unwrap();
        for invalid in [
            pointer(Observation::None, 0, SymbolId(1)),
            pointer(
                Observation::DynamicCandidates(vec![SymbolId(1), SymbolId(1)]),
                0,
                SymbolId(1),
            ),
            pointer(
                Observation::DynamicCandidates(vec![SymbolId(1)]),
                1,
                SymbolId(1),
            ),
            pointer(
                Observation::DynamicCandidates(vec![SymbolId(1)]),
                0,
                SymbolId(2),
            ),
        ] {
            assert!(CausalSequence::from_training(vec![], tokenizer(), &invalid).is_err());
        }
        let mut non_pointer = basic();
        non_pointer.records[0].observation = Observation::DynamicCandidates(vec![SymbolId(1)]);
        assert_eq!(
            convert(&non_pointer),
            Err(SequenceError::MalformedRoute { record: 0 })
        );
    }

    #[test]
    fn annotation_is_not_model_input_or_supervision() {
        let sequence = convert(&basic()).unwrap();
        let Position::Structural(input) = sequence.positions()[2].input() else {
            panic!("structural position")
        };
        assert_eq!(input.observation(), &Observation::None);
        let supervision = sequence.positions()[2].supervision().unwrap();
        assert_eq!(
            supervision.target(),
            &Target::Fixed(FixedTarget::Root(RootAction::Module))
        );
        // There is intentionally no annotation accessor on either public type.
        assert_eq!(sequence.positions()[2].source_annotation, Annotation::None);
    }

    #[test]
    fn reserved_positions_and_english_after_structure_are_rejected() {
        let mut sequence = convert(&basic()).unwrap();
        sequence.positions.push(SequencePosition {
            input: Position::SymbolNameToken(1),
            supervision: None,
            source_annotation: Annotation::None,
        });
        assert_eq!(
            sequence.validate(),
            Err(SequenceError::ReservedPositionInV1 { position: 4 })
        );
        sequence.positions[4].input = Position::FreeToken(1);
        assert_eq!(
            sequence.validate(),
            Err(SequenceError::ReservedPositionInV1 { position: 4 })
        );
        sequence.positions[4].input = Position::EnglishToken(1);
        assert_eq!(
            sequence.validate(),
            Err(SequenceError::EnglishAfterStructural { position: 4 })
        );
    }

    #[test]
    fn metadata_support_is_distinct_from_batch_homogeneity() {
        assert!(TokenizerIdentity::new("", "r", "v").is_err());
        assert!(TokenizerIdentity::new("id", "", "v").is_err());
        assert!(TokenizerIdentity::new("id", "r", "").is_err());
        let sequence = convert(&basic()).unwrap();
        let supported = SupportedMetadata {
            protocol_version: SEQUENCE_PROTOCOL_VERSION,
            training_schema_version: TRAINING_SCHEMA_VERSION,
            tokenizer: tokenizer(),
        };
        sequence.validate_supported(&supported).unwrap();
        let mut wrong = supported.clone();
        wrong.protocol_version += 1;
        assert_eq!(
            sequence.validate_supported(&wrong),
            Err(SequenceError::UnsupportedProtocolVersion)
        );
        let mut wrong = supported.clone();
        wrong.training_schema_version = SchemaVersion(999);
        assert_eq!(
            sequence.validate_supported(&wrong),
            Err(SequenceError::UnsupportedTrainingSchema)
        );
        let mut wrong = supported;
        wrong.tokenizer = TokenizerIdentity::new("other", "r", "v").unwrap();
        assert_eq!(
            sequence.validate_supported(&wrong),
            Err(SequenceError::UnsupportedTokenizer)
        );

        let same = convert(&basic()).unwrap();
        validate_batch(&[sequence.clone(), same]).unwrap();
        let different = CausalSequence::from_training(
            vec![],
            TokenizerIdentity::new("other", "r", "v").unwrap(),
            &basic(),
        )
        .unwrap();
        assert_eq!(
            validate_batch(&[sequence, different]),
            Err(SequenceError::MetadataMismatch { sequence: 1 })
        );
    }

    #[test]
    fn homogeneous_unsupported_metadata_is_still_a_valid_batch() {
        let expected = SupportedMetadata {
            protocol_version: SEQUENCE_PROTOCOL_VERSION,
            training_schema_version: TRAINING_SCHEMA_VERSION,
            tokenizer: tokenizer(),
        };

        let mut unsupported_protocol = convert(&basic()).unwrap();
        unsupported_protocol.metadata.protocol_version = 77;
        validate_batch(&[unsupported_protocol.clone(), unsupported_protocol.clone()]).unwrap();
        assert_eq!(
            unsupported_protocol.validate_supported(&expected),
            Err(SequenceError::UnsupportedProtocolVersion)
        );

        let mut unsupported_schema = convert(&basic()).unwrap();
        unsupported_schema.metadata.training_schema_version = SchemaVersion(77);
        validate_batch(&[unsupported_schema.clone(), unsupported_schema.clone()]).unwrap();
        assert_eq!(
            unsupported_schema.validate_supported(&expected),
            Err(SequenceError::UnsupportedTrainingSchema)
        );

        let mut unsupported_tokenizer = convert(&basic()).unwrap();
        unsupported_tokenizer.metadata.tokenizer =
            TokenizerIdentity::new("different", "r", "v").unwrap();
        validate_batch(&[unsupported_tokenizer.clone(), unsupported_tokenizer.clone()]).unwrap();
        assert_eq!(
            unsupported_tokenizer.validate_supported(&expected),
            Err(SequenceError::UnsupportedTokenizer)
        );

        assert_eq!(
            validate_batch(&[unsupported_schema, unsupported_tokenizer]),
            Err(SequenceError::MetadataMismatch { sequence: 1 })
        );
    }
}
