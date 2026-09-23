//! Stateful fixed-head generation.
//!
//! The runner owns the controller and every model input/action pair, so callers
//! cannot accidentally score one controller state and apply the result to
//! another.

use candle_core::Tensor;
use std::{collections::HashMap, error::Error, fmt};

use crate::{
    controller::{
        Action, BinaryOperatorAction, BlockAction, DeclarationAction, ExpressionAction,
        ItemListAction, ParameterListAction, RootAction, SymbolReferenceAction, TypeAction,
    },
    decode::{ApplyOutcome, Decision, DeclarationKind, DecodeController, DecodeError},
    model::{
        bridge::{
            BridgeError, CandidateSet, ModelRoute, ModelSelection, ModelStep, ModelValue, step,
        },
        decoder::symbol_pointer_scores,
        ir::{Root, SymbolId},
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
    MissingDeclarationMemory(SymbolId),
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
    hidden: Tensor,
    logits: Option<Tensor>,
}

/// A generation session with declaration-keyed dynamic pointer memory.
pub struct GenerationRunner {
    model: StructuralModel,
    controller: DecodeController,
    english_prefix: Vec<TokenId>,
    tokenizer: TokenizerIdentity,
    history: Vec<(StructuralPosition, Action)>,
    declaration_memory: HashMap<SymbolId, Tensor>,
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
            declaration_memory: HashMap::new(),
            current: None,
        }
    }

    /// Observes and scores the current decision, or returns its cached logits.
    pub fn fixed_logits(&mut self) -> Result<&Tensor, RunnerError> {
        self.ensure_current()?;
        let current = self.current.as_ref().expect("initialized above");
        let route = current.step.route().ok_or(BridgeError::Complete)?;
        if !matches!(route, ModelRoute::Fixed(_)) {
            return Err(RunnerError::Unsupported(route));
        }
        Ok(current.logits.as_ref().expect("fixed route has logits"))
    }

    /// Scores the current query against declaration states in controller order.
    pub fn pointer_logits(&mut self) -> Result<Tensor, RunnerError> {
        self.ensure_current()?;
        let current = self.current.as_ref().expect("initialized above");
        let CandidateSet::Symbols(symbols) =
            current.step.candidates().ok_or(BridgeError::Complete)?
        else {
            return Err(RunnerError::Unsupported(
                current.step.route().ok_or(BridgeError::Complete)?,
            ));
        };
        let keys = symbols
            .iter()
            .map(|symbol| {
                self.declaration_memory
                    .get(symbol)
                    .cloned()
                    .ok_or(RunnerError::MissingDeclarationMemory(*symbol))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let refs = keys.iter().collect::<Vec<_>>();
        let keys = Tensor::stack(&refs, 1)?;
        let legal = Tensor::ones((1, symbols.len()), candle_core::DType::U8, keys.device())?;
        Ok(symbol_pointer_scores(&current.hidden, &keys, &legal)?)
    }

    /// Returns the exact ordered symbol slots used by `pointer_logits`.
    pub fn pointer_symbols(&mut self) -> Result<Vec<SymbolId>, RunnerError> {
        self.ensure_current()?;
        match self.current.as_ref().unwrap().step.candidates() {
            Some(CandidateSet::Symbols(symbols)) => Ok(symbols.clone()),
            _ => Err(RunnerError::Unsupported(
                self.current
                    .as_ref()
                    .unwrap()
                    .step
                    .route()
                    .ok_or(BridgeError::Complete)?,
            )),
        }
    }

    /// Validates a stable fixed-head ID against the exact step just scored.
    pub fn apply_fixed(&mut self, head: OutputHead, id: u16) -> Result<(), RunnerError> {
        let current = self.current.as_ref().ok_or(RunnerError::NoCurrentStep)?;
        let decision = current
            .step
            .decision_for(ModelSelection::Fixed { head, id })?;
        let outcome = self.controller.apply(decision)?;
        if let ApplyOutcome::Allocated { symbol, .. } = outcome {
            self.declaration_memory
                .insert(symbol, current.hidden.clone());
        }
        let action = completed_action(decision, outcome)?;
        let current = self.current.take().expect("validated above");
        self.history.push((current.input, action));
        Ok(())
    }

    /// Applies a slot from the exact dynamic-pointer step currently cached.
    pub fn apply_pointer(&mut self, slot: usize) -> Result<(), RunnerError> {
        let current = self.current.as_ref().ok_or(RunnerError::NoCurrentStep)?;
        let decision = current
            .step
            .decision_for(ModelSelection::PointerSlot(slot))?;
        let outcome = self.controller.apply(decision)?;
        let action = completed_action(decision, outcome)?;
        let current = self.current.take().expect("validated above");
        self.history.push((current.input, action));
        Ok(())
    }

    /// Applies a typed noncategorical value to the exact cached request.
    pub fn apply_value(&mut self, value: ModelValue) -> Result<(), RunnerError> {
        self.ensure_current()?;
        let current = self.current.as_ref().expect("initialized above");
        let decision = current.step.value_decision(value)?;
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

    fn ensure_current(&mut self) -> Result<(), RunnerError> {
        if self.current.is_some() {
            return Ok(());
        }
        let model_step = step(&self.controller);
        let route = model_step.route().ok_or(BridgeError::Complete)?;
        let previous = self
            .history
            .last()
            .map_or(PreviousAction::Bos, |(_, action)| {
                PreviousAction::Action(action.clone())
            });
        let input = StructuralPosition::from_model_step(&model_step, previous, self.history.len())?;
        let sequence = CausalSequence::from_inference(
            &self.english_prefix,
            &self.tokenizer,
            &self.history,
            input.clone(),
        );
        let hidden = self.model.last_hidden(&sequence)?;
        let logits = match route {
            ModelRoute::Fixed(_) => {
                Some(self.model.fixed_logits_from_hidden(&hidden, &model_step)?)
            }
            _ => None,
        };
        self.current = Some(Current {
            step: model_step,
            input,
            hidden,
            logits,
        });
        Ok(())
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
