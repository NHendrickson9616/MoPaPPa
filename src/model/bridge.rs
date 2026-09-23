//! The tight boundary between live decoding and model sampling.
//!
//! Call [`step`] immediately before sampling so that the model scores only the
//! returned candidates, then call [`apply_selection`] (or [`apply_value`] for
//! an integer) immediately afterward.  In particular, symbol identities are
//! never model classes: the model scores slots in the ordered, live pointer
//! mask.

use crate::{
    controller::{
        BinaryOperatorAction, BlockAction, ExpressionAction, ItemListAction, Need,
        ParameterListAction, RootAction, TypeAction,
    },
    decode::{
        BlockChoice, ControllerObservation, Decision, DeclarationKind as DecodeDeclarationKind,
        DecodeController, DecodeError, DecodeStateFeatures, DecodeStatus, ExpressionChoice,
        LegalCandidates, ListChoice, RootChoice,
    },
    model::ir::SymbolId,
    training::{DeclarationKind, FixedTarget, OutputHead},
};

/// The model route used for a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelRoute {
    /// A stable, fixed-vocabulary output head.
    Fixed(OutputHead),
    /// A pointer distribution over the accompanying ordered symbol slots.
    DynamicPointer,
    /// A noncategorical value prediction.
    Value(ValueRequest),
}

/// A request that cannot be represented by a finite candidate vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueRequest {
    /// Supply an arbitrary signed 128-bit integer.
    Integer,
}

/// One fixed candidate. `id` is the stable, head-local training-schema ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedCandidate {
    pub id: u16,
    decision: Decision,
}

/// Exactly the distribution that is legal to score at this point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CandidateSet {
    /// Ordered, masked candidates from a fixed training output head.
    Fixed(Vec<FixedCandidate>),
    /// Ordered slots for a dynamic pointer distribution.
    Symbols(Vec<SymbolId>),
    /// A noncategorical prediction request (and therefore no fake candidates).
    Value(ValueRequest),
}

/// Model-facing view of the live controller.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelStep {
    /// A decision is required.
    Needs {
        need: Need,
        route: ModelRoute,
        state: DecodeStateFeatures,
        candidates: CandidateSet,
    },
    /// Decoding has completed; no distribution may be sampled.
    Complete,
}

impl ModelStep {
    /// Required controller category, if decoding is not complete.
    #[must_use]
    pub fn need(&self) -> Option<Need> {
        match self {
            Self::Needs { need, .. } => Some(*need),
            Self::Complete => None,
        }
    }

    /// Model output route, if decoding is not complete.
    #[must_use]
    pub fn route(&self) -> Option<ModelRoute> {
        match self {
            Self::Needs { route, .. } => Some(*route),
            Self::Complete => None,
        }
    }

    /// Compact controller features, if decoding is not complete.
    #[must_use]
    pub fn state(&self) -> Option<DecodeStateFeatures> {
        match self {
            Self::Needs { state, .. } => Some(*state),
            Self::Complete => None,
        }
    }

    /// Legal candidates (or value request), if decoding is not complete.
    #[must_use]
    pub fn candidates(&self) -> Option<&CandidateSet> {
        match self {
            Self::Needs { candidates, .. } => Some(candidates),
            Self::Complete => None,
        }
    }

    /// Validates a categorical selection against this exact pre-sampling mask.
    pub fn decision_for(&self, selection: ModelSelection) -> Result<Decision, BridgeError> {
        match (self, selection) {
            (Self::Complete, _) => Err(BridgeError::Complete),
            (
                Self::Needs {
                    route: ModelRoute::Fixed(expected),
                    candidates: CandidateSet::Fixed(candidates),
                    ..
                },
                ModelSelection::Fixed { head, id },
            ) => {
                if head != *expected {
                    return Err(BridgeError::WrongHead {
                        expected: *expected,
                        received: head,
                    });
                }
                candidates
                    .iter()
                    .find(|candidate| candidate.id == id)
                    .map(|candidate| candidate.decision)
                    .ok_or(BridgeError::CandidateIdNotLegal { head, id })
            }
            (
                Self::Needs {
                    route: ModelRoute::DynamicPointer,
                    candidates: CandidateSet::Symbols(symbols),
                    ..
                },
                ModelSelection::PointerSlot(slot),
            ) => symbols.get(slot).copied().map(Decision::Symbol).ok_or(
                BridgeError::SlotOutOfRange {
                    slot,
                    candidate_count: symbols.len(),
                },
            ),
            (Self::Needs { route, .. }, received) => Err(BridgeError::WrongRoute {
                expected: *route,
                received: received.route(),
            }),
        }
    }

    /// Supplies a value only when this step explicitly requests one.
    pub fn value_decision(&self, value: ModelValue) -> Result<Decision, BridgeError> {
        match (self, value) {
            (
                Self::Needs {
                    route: ModelRoute::Value(ValueRequest::Integer),
                    candidates: CandidateSet::Value(ValueRequest::Integer),
                    ..
                },
                ModelValue::Integer(value),
            ) => Ok(Decision::Integer(value)),
            (Self::Complete, _) => Err(BridgeError::Complete),
            (Self::Needs { route, .. }, _) => Err(BridgeError::WrongRoute {
                expected: *route,
                received: ModelRoute::Value(ValueRequest::Integer),
            }),
        }
    }
}

/// A categorical model result. Indices and slots are always checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelSelection {
    /// A stable head-local class ID, not an index into the compact legal mask.
    Fixed {
        head: OutputHead,
        id: u16,
    },
    PointerSlot(usize),
}

/// A typed noncategorical model result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelValue {
    Integer(i128),
}

impl ModelSelection {
    fn route(self) -> ModelRoute {
        match self {
            Self::Fixed { head, .. } => ModelRoute::Fixed(head),
            Self::PointerSlot(_) => ModelRoute::DynamicPointer,
        }
    }
}

/// Failure while validating or applying a bridge result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BridgeError {
    Complete,
    WrongHead {
        expected: OutputHead,
        received: OutputHead,
    },
    WrongRoute {
        expected: ModelRoute,
        received: ModelRoute,
    },
    CandidateIdNotLegal {
        head: OutputHead,
        id: u16,
    },
    SlotOutOfRange {
        slot: usize,
        candidate_count: usize,
    },
    Controller(DecodeError),
}

impl From<DecodeError> for BridgeError {
    fn from(value: DecodeError) -> Self {
        Self::Controller(value)
    }
}

/// Observes `controller` and builds its exact pre-sampling model distribution.
#[must_use]
pub fn step(controller: &DecodeController) -> ModelStep {
    match controller.observe() {
        DecodeStatus::Complete => ModelStep::Complete,
        DecodeStatus::Needs(observation) => from_observation(observation),
    }
}

/// Validates and applies one categorical selection, then observes the next step.
pub fn apply_selection(
    controller: &mut DecodeController,
    model_step: &ModelStep,
    selection: ModelSelection,
) -> Result<DecodeStatus, BridgeError> {
    let decision = model_step.decision_for(selection)?;
    controller.apply(decision)?;
    Ok(controller.observe())
}

/// Applies an explicitly supplied integer value, then observes the next step.
pub fn apply_value(
    controller: &mut DecodeController,
    model_step: &ModelStep,
    value: ModelValue,
) -> Result<DecodeStatus, BridgeError> {
    let decision = model_step.value_decision(value)?;
    controller.apply(decision)?;
    Ok(controller.observe())
}

fn from_observation(observation: ControllerObservation) -> ModelStep {
    let (route, candidates) = candidates(observation.legal);
    ModelStep::Needs {
        need: observation.need,
        route,
        state: observation.state,
        candidates,
    }
}

fn candidates(legal: LegalCandidates) -> (ModelRoute, CandidateSet) {
    let fixed = |head, values| (ModelRoute::Fixed(head), CandidateSet::Fixed(values));
    match legal {
        LegalCandidates::Root(values) => fixed(
            OutputHead::Root,
            values
                .into_iter()
                .map(|value| candidate(root_id(value), Decision::Root(value)))
                .collect(),
        ),
        LegalCandidates::ItemList(values) => fixed(
            OutputHead::ItemList,
            values
                .into_iter()
                .map(|value| candidate(list_id(value), Decision::ItemList(value)))
                .collect(),
        ),
        LegalCandidates::Declaration(values) => fixed(
            OutputHead::DeclarationKind,
            values
                .into_iter()
                .map(|value| candidate(declaration_id(value), Decision::Declare(value)))
                .collect(),
        ),
        LegalCandidates::ParameterList(values) => fixed(
            OutputHead::ParameterList,
            values
                .into_iter()
                .map(|value| candidate(parameter_list_id(value), Decision::ParameterList(value)))
                .collect(),
        ),
        LegalCandidates::Type(values) => fixed(
            OutputHead::Type,
            values
                .into_iter()
                .map(|value| {
                    candidate(
                        FixedTarget::Type(TypeAction::Primitive(value)).candidate_id(),
                        Decision::Type(value),
                    )
                })
                .collect(),
        ),
        LegalCandidates::Block(values) => fixed(
            OutputHead::Block,
            values
                .into_iter()
                .map(|value| candidate(block_id(value), Decision::Block(value)))
                .collect(),
        ),
        LegalCandidates::Expression(values) => fixed(
            OutputHead::Expression,
            values
                .into_iter()
                .map(|value| candidate(expression_id(value), Decision::Expression(value)))
                .collect(),
        ),
        LegalCandidates::Integer => (
            ModelRoute::Value(ValueRequest::Integer),
            CandidateSet::Value(ValueRequest::Integer),
        ),
        LegalCandidates::BinaryOperator(values) => fixed(
            OutputHead::BinaryOperator,
            values
                .into_iter()
                .map(|value| {
                    candidate(
                        FixedTarget::BinaryOperator(BinaryOperatorAction::Operator(value))
                            .candidate_id(),
                        Decision::BinaryOperator(value),
                    )
                })
                .collect(),
        ),
        LegalCandidates::Symbols(values) => {
            (ModelRoute::DynamicPointer, CandidateSet::Symbols(values))
        }
    }
}

const fn candidate(id: u16, decision: Decision) -> FixedCandidate {
    FixedCandidate { id, decision }
}

fn root_id(value: RootChoice) -> u16 {
    FixedTarget::Root(match value {
        RootChoice::Module => RootAction::Module,
        RootChoice::BlockFragment => RootAction::BlockFragment,
    })
    .candidate_id()
}

fn list_id(value: ListChoice) -> u16 {
    FixedTarget::ItemList(match value {
        ListChoice::Continue => ItemListAction::Function,
        ListChoice::End => ItemListAction::End,
    })
    .candidate_id()
}

fn parameter_list_id(value: ListChoice) -> u16 {
    FixedTarget::ParameterList(match value {
        ListChoice::Continue => ParameterListAction::Parameter,
        ListChoice::End => ParameterListAction::End,
    })
    .candidate_id()
}

fn declaration_id(value: DecodeDeclarationKind) -> u16 {
    match value {
        DecodeDeclarationKind::Function => DeclarationKind::Function,
        DecodeDeclarationKind::Parameter => DeclarationKind::Parameter,
    }
    .candidate_id()
}

fn block_id(value: BlockChoice) -> u16 {
    FixedTarget::Block(match value {
        BlockChoice::ExpressionStatement => BlockAction::ExpressionStatement,
        BlockChoice::EndAndYield => BlockAction::EndAndYield,
        BlockChoice::EndAndDiscard => BlockAction::EndAndDiscard,
    })
    .candidate_id()
}

fn expression_id(value: ExpressionChoice) -> u16 {
    FixedTarget::Expression(match value {
        ExpressionChoice::Symbol => ExpressionAction::Symbol,
        ExpressionChoice::Integer => ExpressionAction::Literal,
        ExpressionChoice::Binary => ExpressionAction::Binary,
    })
    .candidate_id()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed(step: &ModelStep) -> &[FixedCandidate] {
        match step.candidates() {
            Some(CandidateSet::Fixed(values)) => values,
            other => panic!("expected fixed candidates, got {other:?}"),
        }
    }

    #[test]
    fn exact_observe_step_select_apply_loop_and_complete() {
        let mut controller = DecodeController::new();
        let first = step(&controller);
        assert_eq!(
            fixed(&first).iter().map(|v| v.id).collect::<Vec<_>>(),
            [0, 1]
        );
        apply_selection(
            &mut controller,
            &first,
            ModelSelection::Fixed {
                head: OutputHead::Root,
                id: 1,
            },
        )
        .unwrap();
        let block = step(&controller);
        apply_selection(
            &mut controller,
            &block,
            ModelSelection::Fixed {
                head: OutputHead::Block,
                id: 3,
            },
        )
        .unwrap();
        assert_eq!(step(&controller), ModelStep::Complete);
        assert_eq!(
            ModelStep::Complete.decision_for(ModelSelection::PointerSlot(0)),
            Err(BridgeError::Complete)
        );
    }

    #[test]
    fn rejects_wrong_head_id_and_slot() {
        let controller = DecodeController::new();
        let root = step(&controller);
        assert!(matches!(
            root.decision_for(ModelSelection::Fixed {
                head: OutputHead::Block,
                id: 0
            }),
            Err(BridgeError::WrongHead { .. })
        ));
        assert!(matches!(
            root.decision_for(ModelSelection::Fixed {
                head: OutputHead::Root,
                id: 2
            }),
            Err(BridgeError::CandidateIdNotLegal { .. })
        ));
        assert!(matches!(
            root.decision_for(ModelSelection::PointerSlot(0)),
            Err(BridgeError::WrongRoute { .. })
        ));
    }

    #[test]
    fn integer_is_noncategorical() {
        let mut controller = DecodeController::new();
        let root = step(&controller);
        apply_selection(
            &mut controller,
            &root,
            ModelSelection::Fixed {
                head: OutputHead::Root,
                id: 1,
            },
        )
        .unwrap();
        let block = step(&controller);
        apply_selection(
            &mut controller,
            &block,
            ModelSelection::Fixed {
                head: OutputHead::Block,
                id: 1,
            },
        )
        .unwrap();
        let expression = step(&controller);
        apply_selection(
            &mut controller,
            &expression,
            ModelSelection::Fixed {
                head: OutputHead::Expression,
                id: 1,
            },
        )
        .unwrap();
        let value = step(&controller);
        assert_eq!(
            value.candidates(),
            Some(&CandidateSet::Value(ValueRequest::Integer))
        );
        assert_eq!(
            value.value_decision(ModelValue::Integer(i128::MIN)),
            Ok(Decision::Integer(i128::MIN))
        );
    }

    #[test]
    fn sparse_fixed_ids_are_full_head_classes() {
        let (route, declaration) = candidates(LegalCandidates::Declaration(vec![
            DecodeDeclarationKind::Parameter,
        ]));
        assert_eq!(route, ModelRoute::Fixed(OutputHead::DeclarationKind));
        assert_eq!(
            declaration,
            CandidateSet::Fixed(vec![candidate(
                1,
                Decision::Declare(DecodeDeclarationKind::Parameter)
            )])
        );

        let (_, block) = candidates(LegalCandidates::Block(vec![
            BlockChoice::ExpressionStatement,
            BlockChoice::EndAndYield,
            BlockChoice::EndAndDiscard,
        ]));
        assert_eq!(
            block,
            CandidateSet::Fixed(vec![
                candidate(1, Decision::Block(BlockChoice::ExpressionStatement)),
                candidate(2, Decision::Block(BlockChoice::EndAndYield)),
                candidate(3, Decision::Block(BlockChoice::EndAndDiscard)),
            ])
        );

        let (_, expression) = candidates(LegalCandidates::Expression(vec![
            ExpressionChoice::Integer,
            ExpressionChoice::Binary,
        ]));
        assert_eq!(
            expression,
            CandidateSet::Fixed(vec![
                candidate(1, Decision::Expression(ExpressionChoice::Integer)),
                candidate(2, Decision::Expression(ExpressionChoice::Binary)),
            ])
        );
    }

    #[test]
    fn masked_fixed_id_is_rejected_without_controller_mutation() {
        let mut controller = DecodeController::new();
        let before = controller.observe();
        let root = step(&controller);

        assert_eq!(
            apply_selection(
                &mut controller,
                &root,
                ModelSelection::Fixed {
                    head: OutputHead::Root,
                    id: 7,
                },
            ),
            Err(BridgeError::CandidateIdNotLegal {
                head: OutputHead::Root,
                id: 7,
            })
        );
        assert_eq!(controller.observe(), before);
    }
}
