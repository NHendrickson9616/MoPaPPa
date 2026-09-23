//! Minimal live decoder for the phase-1 generation grammar.

use crate::{
    controller::Need,
    model::ir::{
        BinaryOperator, Block, Expression, Function, Item, Literal, Parameter, PrimitiveType, Root,
        Statement, SymbolId,
    },
};

/// Root choices supported by this decoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootChoice {
    /// Generate a module.
    Module,
    /// Generate a standalone block.
    BlockFragment,
}

/// Explicit list control.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ListChoice {
    /// Add one element.
    Continue,
    /// Finish the list.
    End,
}

/// Ways a block may advance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockChoice {
    /// Add an expression statement.
    ExpressionStatement,
    /// Finish with a tail expression.
    EndAndYield,
    /// Finish without a tail expression.
    EndAndDiscard,
}

/// Expression forms supported in phase 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpressionChoice {
    /// A visible symbol.
    Symbol,
    /// An integer literal.
    Integer,
    /// A binary expression.
    Binary,
}

/// The kind of declaration allocated by the controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeclarationKind {
    /// A free function.
    Function,
    /// A function parameter.
    Parameter,
}

/// A model decision. Unlike [`crate::controller::Action`], declarations carry no ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Select the root.
    Root(RootChoice),
    /// Continue or end the module item list.
    ItemList(ListChoice),
    /// Allocate the pending declaration.
    Declare(DeclarationKind),
    /// Continue or end the parameter list.
    ParameterList(ListChoice),
    /// Select a primitive type.
    Type(PrimitiveType),
    /// Add an entry or end a block.
    Block(BlockChoice),
    /// Select an expression form.
    Expression(ExpressionChoice),
    /// Supply an integer literal.
    Integer(i128),
    /// Select a binary operator.
    BinaryOperator(BinaryOperator),
    /// Select an in-scope symbol.
    Symbol(SymbolId),
}

/// Exact legal candidates for the current decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LegalCandidates {
    /// Legal roots.
    Root(Vec<RootChoice>),
    /// Legal item-list controls.
    ItemList(Vec<ListChoice>),
    /// The sole legal declaration kind.
    Declaration(Vec<DeclarationKind>),
    /// Legal parameter-list controls.
    ParameterList(Vec<ListChoice>),
    /// Legal primitive types.
    Type(Vec<PrimitiveType>),
    /// Legal block controls.
    Block(Vec<BlockChoice>),
    /// Legal expression forms.
    Expression(Vec<ExpressionChoice>),
    /// Integer values are supplied by the model and are not enumerated.
    Integer,
    /// Legal binary operators.
    BinaryOperator(Vec<BinaryOperator>),
    /// Ordered visible symbol IDs.
    Symbols(Vec<SymbolId>),
}

/// The decoder's explicit structural phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodePhase {
    /// Selecting the root kind.
    Root,
    /// Allocating the complete module function header list.
    ModuleHeaders,
    /// Constructing declared functions in header order.
    FunctionBodies,
    /// Constructing a standalone block fragment.
    BlockFragment,
}

/// Small, target-free features describing decoder progress.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeStateFeatures {
    /// Current structural phase.
    pub phase: DecodePhase,
    /// Number of IDs allocated so far.
    pub allocated_symbols: u32,
    /// Number of symbols visible at this position.
    pub visible_symbols: u32,
    /// Current binary-expression nesting depth.
    pub expression_depth: u32,
}

/// What the model sees before choosing a decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControllerObservation {
    /// Required action category.
    pub need: Need,
    /// Compact progress features.
    pub state: DecodeStateFeatures,
    /// Exact legal candidates.
    pub legal: LegalCandidates,
}

/// Whether decoding needs another decision or has completed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeStatus {
    /// Another decision is required.
    Needs(ControllerObservation),
    /// The root is complete.
    Complete,
}

/// Result of applying one decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApplyOutcome {
    /// The decision was accepted.
    Accepted,
    /// A declaration was allocated.
    Allocated {
        /// Newly allocated ID.
        symbol: SymbolId,
        /// Kind of declaration.
        kind: DeclarationKind,
    },
}

/// A rejected decoder operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// The decision belongs to a different action category.
    WrongCategory {
        /// Category currently required.
        expected: Need,
        /// Category supplied.
        received: Need,
    },
    /// The decision is in the right category but is not currently legal.
    IllegalCandidate {
        /// Category in which the candidate was rejected.
        need: Need,
    },
    /// A symbol is not visible at this position.
    UnknownSymbol {
        /// Rejected symbol.
        symbol: SymbolId,
    },
    /// Decoding has already completed.
    AlreadyComplete,
    /// The tree is not complete yet.
    Incomplete {
        /// Category still required.
        need: Need,
    },
}

#[derive(Clone, Debug)]
enum State {
    Root,
    ModuleHeaders(Vec<SymbolId>),
    FunctionDeclaration(Vec<SymbolId>),
    Parameters(FunctionBuild),
    ParameterDeclaration(FunctionBuild),
    ParameterType(FunctionBuild, SymbolId),
    ReturnType(FunctionBuild),
    Block(BlockBuild),
    Expression(ExprBuild),
    Integer(ExprBuild),
    Operator(ExprBuild),
    Symbol(ExprBuild),
    Complete,
}

#[derive(Clone, Debug)]
struct FunctionBuild {
    headers: Vec<SymbolId>,
    header_index: usize,
    items: Vec<Item>,
    name: SymbolId,
    parameters: Vec<Parameter>,
    visible: Vec<SymbolId>,
}

#[derive(Clone, Debug)]
struct BlockBuild {
    statements: Vec<Statement>,
    visible: Vec<SymbolId>,
    done: BlockDone,
}

#[derive(Clone, Debug)]
enum BlockDone {
    Root,
    Function {
        build: FunctionBuild,
        return_type: PrimitiveType,
    },
}

#[derive(Clone, Debug)]
struct ExprBuild {
    visible: Vec<SymbolId>,
    depth: u32,
    done: ExprDone,
}

#[derive(Clone, Debug)]
enum ExprDone {
    Statement(BlockBuild),
    Tail(BlockBuild),
    BinaryLeft {
        operator: BinaryOperator,
        parent: Box<ExprBuild>,
    },
    BinaryRight {
        left: Expression,
        operator: BinaryOperator,
        parent: Box<ExprBuild>,
    },
}

/// Stateful phase-1 decoder.
#[derive(Clone, Debug)]
pub struct DecodeController {
    state: State,
    next_symbol: u32,
    root: Option<Root>,
}

impl Default for DecodeController {
    fn default() -> Self {
        Self::new()
    }
}

impl DecodeController {
    /// Starts a decoder requiring root selection.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: State::Root,
            next_symbol: 0,
            root: None,
        }
    }

    /// Observes the current need and all legal candidates before sampling.
    #[must_use]
    pub fn observe(&self) -> DecodeStatus {
        if matches!(self.state, State::Complete) {
            return DecodeStatus::Complete;
        }
        let need = self.need();
        let (visible, depth) = match &self.state {
            State::Block(build) => (build.visible.len(), 0),
            State::Expression(build)
            | State::Integer(build)
            | State::Operator(build)
            | State::Symbol(build) => (build.visible.len(), build.depth),
            State::Parameters(build)
            | State::ParameterDeclaration(build)
            | State::ReturnType(build) => (build.visible.len(), 0),
            State::ParameterType(build, _) => (build.visible.len(), 0),
            _ => (0, 0),
        };
        DecodeStatus::Needs(ControllerObservation {
            need,
            state: DecodeStateFeatures {
                phase: self.phase(),
                allocated_symbols: self.next_symbol,
                visible_symbols: visible as u32,
                expression_depth: depth,
            },
            legal: self.legal(),
        })
    }

    /// Applies exactly one currently legal decision.
    pub fn apply(&mut self, decision: Decision) -> Result<ApplyOutcome, DecodeError> {
        if matches!(self.state, State::Complete) {
            return Err(DecodeError::AlreadyComplete);
        }
        let expected = self.need();
        let received = decision.need();
        if expected != received {
            return Err(DecodeError::WrongCategory { expected, received });
        }
        let state = std::mem::replace(&mut self.state, State::Complete);
        self.apply_to(state, decision)
    }

    /// Returns the completed tree.
    pub fn finish(&mut self) -> Result<Root, DecodeError> {
        if !matches!(self.state, State::Complete) {
            return Err(DecodeError::Incomplete { need: self.need() });
        }
        self.root.take().ok_or(DecodeError::AlreadyComplete)
    }

    fn need(&self) -> Need {
        match self.state {
            State::Root => Need::Root,
            State::ModuleHeaders(_) => Need::ItemOrEnd,
            State::FunctionDeclaration(_) | State::ParameterDeclaration(_) => Need::Declaration,
            State::Parameters(_) => Need::ParameterOrEnd,
            State::ParameterType(_, _) | State::ReturnType(_) => Need::Type,
            State::Block(_) => Need::BlockEntryOrEnd,
            State::Expression(_) => Need::Expression,
            State::Integer(_) => Need::Literal,
            State::Operator(_) => Need::BinaryOperator,
            State::Symbol(_) => Need::SymbolReference,
            State::Complete => unreachable!("complete states are observed explicitly"),
        }
    }

    fn phase(&self) -> DecodePhase {
        match &self.state {
            State::Root | State::Complete => DecodePhase::Root,
            State::ModuleHeaders(_) | State::FunctionDeclaration(_) => DecodePhase::ModuleHeaders,
            State::Parameters(_)
            | State::ParameterDeclaration(_)
            | State::ParameterType(_, _)
            | State::ReturnType(_) => DecodePhase::FunctionBodies,
            State::Block(build) => block_phase(build),
            State::Expression(build)
            | State::Integer(build)
            | State::Operator(build)
            | State::Symbol(build) => expression_phase(build),
        }
    }

    fn legal(&self) -> LegalCandidates {
        match &self.state {
            State::Root => {
                LegalCandidates::Root(vec![RootChoice::Module, RootChoice::BlockFragment])
            }
            State::ModuleHeaders(_) => {
                LegalCandidates::ItemList(vec![ListChoice::Continue, ListChoice::End])
            }
            State::FunctionDeclaration(_) => {
                LegalCandidates::Declaration(vec![DeclarationKind::Function])
            }
            State::Parameters(_) => {
                LegalCandidates::ParameterList(vec![ListChoice::Continue, ListChoice::End])
            }
            State::ParameterDeclaration(_) => {
                LegalCandidates::Declaration(vec![DeclarationKind::Parameter])
            }
            State::ParameterType(_, _) | State::ReturnType(_) => {
                LegalCandidates::Type(PRIMITIVE_TYPES.to_vec())
            }
            State::Block(_) => LegalCandidates::Block(vec![
                BlockChoice::ExpressionStatement,
                BlockChoice::EndAndYield,
                BlockChoice::EndAndDiscard,
            ]),
            State::Expression(build) => {
                let mut choices = Vec::with_capacity(3);
                if !build.visible.is_empty() {
                    choices.push(ExpressionChoice::Symbol);
                }
                choices.extend([ExpressionChoice::Integer, ExpressionChoice::Binary]);
                LegalCandidates::Expression(choices)
            }
            State::Integer(_) => LegalCandidates::Integer,
            State::Operator(_) => LegalCandidates::BinaryOperator(BINARY_OPERATORS.to_vec()),
            State::Symbol(build) => LegalCandidates::Symbols(build.visible.clone()),
            State::Complete => unreachable!("complete states have no candidates"),
        }
    }

    fn apply_to(&mut self, state: State, decision: Decision) -> Result<ApplyOutcome, DecodeError> {
        match (state, decision) {
            (State::Root, Decision::Root(RootChoice::Module)) => {
                self.state = State::ModuleHeaders(Vec::new());
            }
            (State::Root, Decision::Root(RootChoice::BlockFragment)) => {
                self.state = State::Block(BlockBuild {
                    statements: Vec::new(),
                    visible: Vec::new(),
                    done: BlockDone::Root,
                });
            }
            (State::ModuleHeaders(headers), Decision::ItemList(ListChoice::Continue)) => {
                self.state = State::FunctionDeclaration(headers);
            }
            (State::ModuleHeaders(headers), Decision::ItemList(ListChoice::End)) => {
                if headers.is_empty() {
                    self.root = Some(Root::Module { items: Vec::new() });
                    self.state = State::Complete;
                } else {
                    self.start_function_body(headers, 0, Vec::new());
                }
            }
            (
                State::FunctionDeclaration(mut headers),
                Decision::Declare(DeclarationKind::Function),
            ) => {
                let symbol = self.allocate();
                headers.push(symbol);
                self.state = State::ModuleHeaders(headers);
                return Ok(ApplyOutcome::Allocated {
                    symbol,
                    kind: DeclarationKind::Function,
                });
            }
            (State::Parameters(build), Decision::ParameterList(ListChoice::Continue)) => {
                self.state = State::ParameterDeclaration(build);
            }
            (State::Parameters(build), Decision::ParameterList(ListChoice::End)) => {
                self.state = State::ReturnType(build);
            }
            (State::ParameterDeclaration(build), Decision::Declare(DeclarationKind::Parameter)) => {
                let symbol = self.allocate();
                self.state = State::ParameterType(build, symbol);
                return Ok(ApplyOutcome::Allocated {
                    symbol,
                    kind: DeclarationKind::Parameter,
                });
            }
            (State::ParameterType(mut build, symbol), Decision::Type(ty)) => {
                build.parameters.push(Parameter { name: symbol, ty });
                build.visible.push(symbol);
                self.state = State::Parameters(build);
            }
            (State::ReturnType(build), Decision::Type(return_type)) => {
                let visible = build.visible.clone();
                self.state = State::Block(BlockBuild {
                    statements: Vec::new(),
                    visible,
                    done: BlockDone::Function { build, return_type },
                });
            }
            (State::Block(build), Decision::Block(BlockChoice::ExpressionStatement)) => {
                self.start_expression(ExprDone::Statement(build));
            }
            (State::Block(build), Decision::Block(BlockChoice::EndAndYield)) => {
                self.start_expression(ExprDone::Tail(build));
            }
            (State::Block(build), Decision::Block(BlockChoice::EndAndDiscard)) => {
                self.complete_block(build, None);
            }
            (State::Expression(build), Decision::Expression(ExpressionChoice::Symbol)) => {
                if build.visible.is_empty() {
                    self.state = State::Expression(build);
                    return Err(DecodeError::IllegalCandidate {
                        need: Need::Expression,
                    });
                }
                self.state = State::Symbol(build);
            }
            (State::Expression(build), Decision::Expression(ExpressionChoice::Integer)) => {
                self.state = State::Integer(build);
            }
            (State::Expression(build), Decision::Expression(ExpressionChoice::Binary)) => {
                self.state = State::Operator(build);
            }
            (State::Integer(build), Decision::Integer(value)) => {
                self.complete_expression(build, Expression::Literal(Literal::Integer(value)));
            }
            (State::Operator(build), Decision::BinaryOperator(operator)) => {
                let visible = build.visible.clone();
                let depth = build.depth + 1;
                self.state = State::Expression(ExprBuild {
                    visible,
                    depth,
                    done: ExprDone::BinaryLeft {
                        operator,
                        parent: Box::new(build),
                    },
                });
            }
            (State::Symbol(build), Decision::Symbol(symbol)) => {
                if !build.visible.contains(&symbol) {
                    self.state = State::Symbol(build);
                    return Err(DecodeError::UnknownSymbol { symbol });
                }
                self.complete_expression(build, Expression::Symbol(symbol));
            }
            (state, _) => {
                self.state = state;
                return Err(DecodeError::IllegalCandidate {
                    need: expected_need(&self.state),
                });
            }
        }
        Ok(ApplyOutcome::Accepted)
    }

    fn allocate(&mut self) -> SymbolId {
        let symbol = SymbolId(self.next_symbol);
        self.next_symbol = self
            .next_symbol
            .checked_add(1)
            .expect("phase-1 symbol ID space exhausted");
        symbol
    }

    fn start_function_body(
        &mut self,
        headers: Vec<SymbolId>,
        header_index: usize,
        items: Vec<Item>,
    ) {
        self.state = State::Parameters(FunctionBuild {
            name: headers[header_index],
            visible: headers.clone(),
            headers,
            header_index,
            items,
            parameters: Vec::new(),
        });
    }

    fn start_expression(&mut self, done: ExprDone) {
        let visible = match &done {
            ExprDone::Statement(block) | ExprDone::Tail(block) => block.visible.clone(),
            ExprDone::BinaryLeft { parent, .. } | ExprDone::BinaryRight { parent, .. } => {
                parent.visible.clone()
            }
        };
        self.state = State::Expression(ExprBuild {
            visible,
            depth: 0,
            done,
        });
    }

    fn complete_expression(&mut self, build: ExprBuild, expression: Expression) {
        match build.done {
            ExprDone::Statement(mut block) => {
                block.statements.push(Statement::Expression(expression));
                self.state = State::Block(block);
            }
            ExprDone::Tail(block) => self.complete_block(block, Some(expression)),
            ExprDone::BinaryLeft { operator, parent } => {
                let visible = parent.visible.clone();
                let depth = parent.depth + 1;
                self.state = State::Expression(ExprBuild {
                    visible,
                    depth,
                    done: ExprDone::BinaryRight {
                        left: expression,
                        operator,
                        parent,
                    },
                });
            }
            ExprDone::BinaryRight {
                left,
                operator,
                parent,
            } => self.complete_expression(
                *parent,
                Expression::Binary {
                    left: Box::new(left),
                    operator,
                    right: Box::new(expression),
                },
            ),
        }
    }

    fn complete_block(&mut self, build: BlockBuild, tail: Option<Expression>) {
        let block = Block {
            statements: build.statements,
            tail: tail.map(Box::new),
        };
        match build.done {
            BlockDone::Root => {
                self.root = Some(Root::BlockFragment(block));
                self.state = State::Complete;
            }
            BlockDone::Function {
                mut build,
                return_type,
            } => {
                build.items.push(Item::Function(Function {
                    name: build.name,
                    parameters: build.parameters,
                    return_type,
                    body: block,
                }));
                let next = build.header_index + 1;
                if next == build.headers.len() {
                    self.root = Some(Root::Module { items: build.items });
                    self.state = State::Complete;
                } else {
                    self.start_function_body(build.headers, next, build.items);
                }
            }
        }
    }
}

impl Decision {
    fn need(self) -> Need {
        match self {
            Self::Root(_) => Need::Root,
            Self::ItemList(_) => Need::ItemOrEnd,
            Self::Declare(_) => Need::Declaration,
            Self::ParameterList(_) => Need::ParameterOrEnd,
            Self::Type(_) => Need::Type,
            Self::Block(_) => Need::BlockEntryOrEnd,
            Self::Expression(_) => Need::Expression,
            Self::Integer(_) => Need::Literal,
            Self::BinaryOperator(_) => Need::BinaryOperator,
            Self::Symbol(_) => Need::SymbolReference,
        }
    }
}

fn block_phase(build: &BlockBuild) -> DecodePhase {
    match build.done {
        BlockDone::Root => DecodePhase::BlockFragment,
        BlockDone::Function { .. } => DecodePhase::FunctionBodies,
    }
}

fn expression_phase(build: &ExprBuild) -> DecodePhase {
    match &build.done {
        ExprDone::Statement(block) | ExprDone::Tail(block) => block_phase(block),
        ExprDone::BinaryLeft { parent, .. } | ExprDone::BinaryRight { parent, .. } => {
            expression_phase(parent)
        }
    }
}

fn expected_need(state: &State) -> Need {
    match state {
        State::Root => Need::Root,
        State::ModuleHeaders(_) => Need::ItemOrEnd,
        State::FunctionDeclaration(_) | State::ParameterDeclaration(_) => Need::Declaration,
        State::Parameters(_) => Need::ParameterOrEnd,
        State::ParameterType(_, _) | State::ReturnType(_) => Need::Type,
        State::Block(_) => Need::BlockEntryOrEnd,
        State::Expression(_) => Need::Expression,
        State::Integer(_) => Need::Literal,
        State::Operator(_) => Need::BinaryOperator,
        State::Symbol(_) => Need::SymbolReference,
        State::Complete => Need::Root,
    }
}

const PRIMITIVE_TYPES: [PrimitiveType; 11] = [
    PrimitiveType::Bool,
    PrimitiveType::Char,
    PrimitiveType::I32,
    PrimitiveType::I64,
    PrimitiveType::Isize,
    PrimitiveType::U32,
    PrimitiveType::U64,
    PrimitiveType::Usize,
    PrimitiveType::F32,
    PrimitiveType::F64,
    PrimitiveType::Unit,
];

const BINARY_OPERATORS: [BinaryOperator; 12] = [
    BinaryOperator::Add,
    BinaryOperator::Subtract,
    BinaryOperator::Multiply,
    BinaryOperator::Divide,
    BinaryOperator::Equal,
    BinaryOperator::NotEqual,
    BinaryOperator::LessThan,
    BinaryOperator::LessOrEqual,
    BinaryOperator::GreaterThan,
    BinaryOperator::GreaterOrEqual,
    BinaryOperator::And,
    BinaryOperator::Or,
];

#[cfg(test)]
mod tests {
    use super::*;

    fn accept(controller: &mut DecodeController, decision: Decision) -> ApplyOutcome {
        controller.apply(decision).expect("advertised decision")
    }

    fn observation(controller: &DecodeController) -> ControllerObservation {
        let DecodeStatus::Needs(observation) = controller.observe() else {
            panic!("decoder unexpectedly complete");
        };
        observation
    }

    fn declare_function(controller: &mut DecodeController) -> SymbolId {
        accept(controller, Decision::ItemList(ListChoice::Continue));
        let ApplyOutcome::Allocated { symbol, kind } =
            accept(controller, Decision::Declare(DeclarationKind::Function))
        else {
            panic!("function declaration did not allocate");
        };
        assert_eq!(kind, DeclarationKind::Function);
        symbol
    }

    fn empty_function(controller: &mut DecodeController) {
        accept(controller, Decision::ParameterList(ListChoice::End));
        accept(controller, Decision::Type(PrimitiveType::Unit));
        accept(controller, Decision::Block(BlockChoice::EndAndDiscard));
    }

    #[test]
    fn empty_module_completes_when_empty_headers_end() {
        let mut controller = DecodeController::new();
        accept(&mut controller, Decision::Root(RootChoice::Module));
        assert_eq!(
            observation(&controller).state.phase,
            DecodePhase::ModuleHeaders
        );
        accept(&mut controller, Decision::ItemList(ListChoice::End));
        assert_eq!(controller.observe(), DecodeStatus::Complete);
        assert_eq!(controller.finish(), Ok(Root::Module { items: vec![] }));
    }

    #[test]
    fn increment_uses_header_then_body_sequence() {
        let mut controller = DecodeController::new();
        assert_eq!(observation(&controller).need, Need::Root);
        accept(&mut controller, Decision::Root(RootChoice::Module));
        assert_eq!(declare_function(&mut controller), SymbolId(0));
        accept(&mut controller, Decision::ItemList(ListChoice::End));
        assert_eq!(
            observation(&controller).state.phase,
            DecodePhase::FunctionBodies
        );
        accept(
            &mut controller,
            Decision::ParameterList(ListChoice::Continue),
        );
        assert_eq!(
            accept(
                &mut controller,
                Decision::Declare(DeclarationKind::Parameter)
            ),
            ApplyOutcome::Allocated {
                symbol: SymbolId(1),
                kind: DeclarationKind::Parameter,
            }
        );
        accept(&mut controller, Decision::Type(PrimitiveType::I32));
        accept(&mut controller, Decision::ParameterList(ListChoice::End));
        accept(&mut controller, Decision::Type(PrimitiveType::I32));
        accept(&mut controller, Decision::Block(BlockChoice::EndAndYield));
        accept(
            &mut controller,
            Decision::Expression(ExpressionChoice::Binary),
        );
        accept(
            &mut controller,
            Decision::BinaryOperator(BinaryOperator::Add),
        );
        accept(
            &mut controller,
            Decision::Expression(ExpressionChoice::Symbol),
        );
        accept(&mut controller, Decision::Symbol(SymbolId(1)));
        accept(
            &mut controller,
            Decision::Expression(ExpressionChoice::Integer),
        );
        accept(&mut controller, Decision::Integer(1));
        assert_eq!(controller.observe(), DecodeStatus::Complete);

        let Root::Module { items } = controller.finish().unwrap() else {
            panic!("expected module");
        };
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn earlier_body_can_reference_later_function() {
        let mut controller = DecodeController::new();
        accept(&mut controller, Decision::Root(RootChoice::Module));
        assert_eq!(declare_function(&mut controller), SymbolId(0));
        assert_eq!(declare_function(&mut controller), SymbolId(1));
        accept(&mut controller, Decision::ItemList(ListChoice::End));

        accept(&mut controller, Decision::ParameterList(ListChoice::End));
        accept(&mut controller, Decision::Type(PrimitiveType::Unit));
        accept(&mut controller, Decision::Block(BlockChoice::EndAndYield));
        accept(
            &mut controller,
            Decision::Expression(ExpressionChoice::Symbol),
        );
        assert_eq!(
            observation(&controller).legal,
            LegalCandidates::Symbols(vec![SymbolId(0), SymbolId(1)])
        );
        accept(&mut controller, Decision::Symbol(SymbolId(1)));
        empty_function(&mut controller);

        let Root::Module { items } = controller.finish().unwrap() else {
            panic!("expected module");
        };
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn symbol_form_is_hidden_without_visible_symbols_and_rejected_without_mutation() {
        let mut controller = DecodeController::new();
        accept(&mut controller, Decision::Root(RootChoice::BlockFragment));
        accept(&mut controller, Decision::Block(BlockChoice::EndAndYield));
        let before = controller.observe();
        assert_eq!(
            observation(&controller).legal,
            LegalCandidates::Expression(vec![ExpressionChoice::Integer, ExpressionChoice::Binary,])
        );
        assert_eq!(
            controller.apply(Decision::Expression(ExpressionChoice::Symbol)),
            Err(DecodeError::IllegalCandidate {
                need: Need::Expression
            })
        );
        assert_eq!(controller.observe(), before);
    }

    #[test]
    fn block_fragment_has_no_symbol_dead_end() {
        let mut controller = DecodeController::new();
        accept(&mut controller, Decision::Root(RootChoice::BlockFragment));
        accept(&mut controller, Decision::Block(BlockChoice::EndAndYield));
        for _ in 0..3 {
            accept(
                &mut controller,
                Decision::Expression(ExpressionChoice::Binary),
            );
            accept(
                &mut controller,
                Decision::BinaryOperator(BinaryOperator::Add),
            );
        }
        assert!(!matches!(
            observation(&controller).legal,
            LegalCandidates::Expression(choices) if choices.is_empty()
        ));
    }

    #[test]
    fn completion_is_explicit_and_apply_afterward_is_rejected() {
        let mut controller = DecodeController::new();
        accept(&mut controller, Decision::Root(RootChoice::BlockFragment));
        accept(&mut controller, Decision::Block(BlockChoice::EndAndDiscard));
        assert_eq!(controller.observe(), DecodeStatus::Complete);
        assert_eq!(
            controller.apply(Decision::Root(RootChoice::Module)),
            Err(DecodeError::AlreadyComplete)
        );
    }

    #[test]
    fn advertised_finite_candidates_are_accepted_without_immediate_dead_ends() {
        let states = {
            let root = DecodeController::new();
            let mut headers = root.clone();
            accept(&mut headers, Decision::Root(RootChoice::Module));
            let mut block = root.clone();
            accept(&mut block, Decision::Root(RootChoice::BlockFragment));
            let mut expression = block.clone();
            accept(&mut expression, Decision::Block(BlockChoice::EndAndYield));
            vec![root, headers, block, expression]
        };

        for state in states {
            let legal = observation(&state).legal.clone();
            let decisions: Vec<Decision> = match legal {
                LegalCandidates::Root(values) => values.into_iter().map(Decision::Root).collect(),
                LegalCandidates::ItemList(values) => {
                    values.into_iter().map(Decision::ItemList).collect()
                }
                LegalCandidates::Block(values) => values.into_iter().map(Decision::Block).collect(),
                LegalCandidates::Expression(values) => {
                    values.into_iter().map(Decision::Expression).collect()
                }
                _ => continue,
            };
            for decision in decisions {
                let mut clone = state.clone();
                assert!(clone.apply(decision).is_ok());
                if let DecodeStatus::Needs(next) = clone.observe() {
                    assert!(!finite_candidates_empty(&next.legal));
                }
            }
        }
    }

    fn finite_candidates_empty(legal: &LegalCandidates) -> bool {
        match legal {
            LegalCandidates::Root(values) => values.is_empty(),
            LegalCandidates::ItemList(values) => values.is_empty(),
            LegalCandidates::Declaration(values) => values.is_empty(),
            LegalCandidates::ParameterList(values) => values.is_empty(),
            LegalCandidates::Type(values) => values.is_empty(),
            LegalCandidates::Block(values) => values.is_empty(),
            LegalCandidates::Expression(values) => values.is_empty(),
            LegalCandidates::BinaryOperator(values) => values.is_empty(),
            LegalCandidates::Symbols(values) => values.is_empty(),
            LegalCandidates::Integer => false,
        }
    }
}
