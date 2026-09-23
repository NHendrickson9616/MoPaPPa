//! Stateful fixed-head generation.
//!
//! The runner owns the controller and every model input/action pair, so callers
//! cannot accidentally score one controller state and apply the result to
//! another.

use candle_core::Tensor;
use std::{error::Error, fmt};

use crate::{
    controller::{
        Action, BinaryOperatorAction, BlockAction, DeclarationAction, ExpressionAction,
        ItemListAction, ParameterListAction, RootAction, SymbolReferenceAction, TypeAction,
    },
    decode::{ApplyOutcome, Decision, DeclarationKind, DecodeController, DecodeError},
    model::{
        bridge::{BridgeError, ModelRoute, ModelSelection, ModelStep, step},
        ir::Root,
        sequence::{
            CausalSequence, PreviousAction, SequenceError, StructuralPosition, TokenId,
            TokenizerIdentity,
        },
        structural::StructuralModel,
    },
    training::OutputHead,
};

/// Failure while scoring or accepting a generation decision.
#[derive(Debug)]
pub enum RunnerError {
    Model(candle_core::Error),
    Sequence(SequenceError),
    Bridge(BridgeError),
    Controller(DecodeError),
    Unsupported(ModelRoute),
    NoCurrentStep,
}

impl fmt::Display for RunnerError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(output, "{self:?}")
    }
}

impl Error for RunnerError {}

impl From<candle_core::Error> for RunnerError {
    fn from(value: candle_core::Error) -> Self {
        Self::Model(value)
    }
}
impl From<SequenceError> for RunnerError {
    fn from(value: SequenceError) -> Self {
        Self::Sequence(value)
    }
}
impl From<BridgeError> for RunnerError {
    fn from(value: BridgeError) -> Self {
        Self::Bridge(value)
    }
}
impl From<DecodeError> for RunnerError {
    fn from(value: DecodeError) -> Self {
        Self::Controller(value)
    }
}

struct Current {
    step: ModelStep,
    input: StructuralPosition,
    logits: Tensor,
}

/// A generation session supporting the model's fixed categorical heads.
pub struct GenerationRunner {
    model: StructuralModel,
    controller: DecodeController,
    english_prefix: Vec<TokenId>,
    tokenizer: TokenizerIdentity,
    history: Vec<(StructuralPosition, Action)>,
    current: Option<Current>,
}

impl GenerationRunner {
    pub fn new(
        model: StructuralModel,
        english_prefix: Vec<TokenId>,
        tokenizer: TokenizerIdentity,
    ) -> Self {
        Self {
            model,
            controller: DecodeController::new(),
            english_prefix,
            tokenizer,
            history: Vec::new(),
            current: None,
        }
    }

    /// Observes and scores the current decision, or returns its cached logits.
    pub fn fixed_logits(&mut self) -> Result<&Tensor, RunnerError> {
        if self.current.is_none() {
            let model_step = step(&self.controller);
            let route = model_step.route().ok_or(BridgeError::Complete)?;
            if !matches!(route, ModelRoute::Fixed(_)) {
                return Err(RunnerError::Unsupported(route));
            }
            let previous = self
                .history
                .last()
                .map_or(PreviousAction::Bos, |(_, action)| {
                    PreviousAction::Action(action.clone())
                });
            let input =
                StructuralPosition::from_model_step(&model_step, previous, self.history.len())?;
            let sequence = CausalSequence::from_inference(
                &self.english_prefix,
                &self.tokenizer,
                &self.history,
                input.clone(),
            );
            let logits = self.model.fixed_logits(&sequence, &model_step)?;
            self.current = Some(Current {
                step: model_step,
                input,
                logits,
            });
        }
        Ok(&self.current.as_ref().expect("initialized above").logits)
    }

    /// Validates a stable fixed-head ID against the exact step just scored.
    pub fn apply_fixed(&mut self, head: OutputHead, id: u16) -> Result<(), RunnerError> {
        let current = self.current.as_ref().ok_or(RunnerError::NoCurrentStep)?;
        let decision = current
            .step
            .decision_for(ModelSelection::Fixed { head, id })?;
        let outcome = self.controller.apply(decision)?;
        let action = completed_action(decision, outcome)?;
        let current = self.current.take().expect("validated above");
        self.history.push((current.input, action));
        Ok(())
    }

    /// Returns the completed IR root.
    pub fn finish(&mut self) -> Result<Root, RunnerError> {
        Ok(self.controller.finish()?)
    }

    pub fn actions(&self) -> impl ExactSizeIterator<Item = &Action> {
        self.history.iter().map(|(_, action)| action)
    }
}

fn completed_action(decision: Decision, outcome: ApplyOutcome) -> Result<Action, RunnerError> {
    let action = match (decision, outcome) {
        (Decision::Root(value), ApplyOutcome::Accepted) => Action::Root(match value {
            crate::decode::RootChoice::Module => RootAction::Module,
            crate::decode::RootChoice::BlockFragment => RootAction::BlockFragment,
        }),
        (Decision::ItemList(value), ApplyOutcome::Accepted) => Action::ItemList(match value {
            crate::decode::ListChoice::Continue => ItemListAction::Function,
            crate::decode::ListChoice::End => ItemListAction::End,
        }),
        (Decision::Declare(_), ApplyOutcome::Allocated { symbol, kind }) => {
            Action::Declaration(match kind {
                DeclarationKind::Function => DeclarationAction::Function(symbol),
                DeclarationKind::Parameter => DeclarationAction::Parameter(symbol),
            })
        }
        (Decision::ParameterList(value), ApplyOutcome::Accepted) => {
            Action::ParameterList(match value {
                crate::decode::ListChoice::Continue => ParameterListAction::Parameter,
                crate::decode::ListChoice::End => ParameterListAction::End,
            })
        }
        (Decision::Type(value), ApplyOutcome::Accepted) => {
            Action::Type(TypeAction::Primitive(value))
        }
        (Decision::Block(value), ApplyOutcome::Accepted) => Action::Block(match value {
            crate::decode::BlockChoice::ExpressionStatement => BlockAction::ExpressionStatement,
            crate::decode::BlockChoice::EndAndYield => BlockAction::EndAndYield,
            crate::decode::BlockChoice::EndAndDiscard => BlockAction::EndAndDiscard,
        }),
        (Decision::Expression(value), ApplyOutcome::Accepted) => Action::Expression(match value {
            crate::decode::ExpressionChoice::Symbol => ExpressionAction::Symbol,
            crate::decode::ExpressionChoice::Integer => ExpressionAction::Literal,
            crate::decode::ExpressionChoice::Binary => ExpressionAction::Binary,
        }),
        (Decision::Integer(value), ApplyOutcome::Accepted) => {
            Action::Literal(crate::controller::LiteralAction::Integer(value))
        }
        (Decision::BinaryOperator(value), ApplyOutcome::Accepted) => {
            Action::BinaryOperator(BinaryOperatorAction::Operator(value))
        }
        (Decision::Symbol(symbol), ApplyOutcome::Accepted) => {
            Action::SymbolReference(SymbolReferenceAction { symbol })
        }
        _ => return Err(RunnerError::Controller(DecodeError::AlreadyComplete)),
    };
    Ok(action)
}
