//! Deterministic structural action traces for the MVP IR.
//!
//! This module derives and validates action traces from completed IR. It does
//! not yet provide a mutable controller or replay engine.

use std::collections::{BTreeMap, BTreeSet};

use crate::model::ir::{
    BinaryOperator, Block, Expression, Item, Literal, PrimitiveType, Root, Statement, SymbolId,
};

/// The category of action required at a trace position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Need {
    /// Select the root rendering context.
    Root,
    /// Select another module item or finish the item list.
    ItemOrEnd,
    /// Allocate a declared symbol identity.
    Declaration,
    /// Select another function parameter or finish the parameter list.
    ParameterOrEnd,
    /// Select a primitive type.
    Type,
    /// Select a block entry or one of its two end modes.
    BlockEntryOrEnd,
    /// Select whether a `let` has an explicit type annotation.
    TypeAnnotation,
    /// Select an expression form.
    Expression,
    /// Select a literal value.
    Literal,
    /// Select a binary operator.
    BinaryOperator,
    /// Point to a visible symbol.
    SymbolReference,
    /// Point to a visible function for a direct call.
    DirectCallTarget,
    /// Select another call argument or finish the argument list.
    CallArgumentOrEnd,
    /// Select whether an `if` has an `else` block.
    IfElse,
}

/// A root-context action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootAction {
    /// Top-level item output.
    Module,
    /// A block-expression fragment.
    BlockFragment,
}

/// A module item-list action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItemListAction {
    /// Emit a function item.
    Function,
    /// Finish the item list.
    End,
}

/// A declaration allocation action.
///
/// The completed IR supplies the stable ID here. A future controller will
/// allocate the ID while applying this action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeclarationAction {
    /// Declare a module function.
    Function(SymbolId),
    /// Declare a function parameter.
    Parameter(SymbolId),
    /// Declare a block-local binding.
    ///
    /// The ID is allocated before its initializer is traced, but is not visible
    /// to that initializer. It becomes visible to following block entries.
    Local(SymbolId),
}

/// A function parameter-list action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterListAction {
    /// Emit another parameter.
    Parameter,
    /// Finish the parameter list.
    End,
}

/// A primitive-type action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypeAction {
    /// Emit a primitive type.
    Primitive(PrimitiveType),
}

/// A block entry or end action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockAction {
    /// Start a `let` statement.
    Let,
    /// Start a semicolon-terminated expression statement.
    ExpressionStatement,
    /// Finish the block with its pending tail expression.
    EndAndYield,
    /// Finish the block with unit.
    EndAndDiscard,
}

/// A local type-annotation action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypeAnnotationAction {
    /// Emit an explicit annotation; a [`Need::Type`] step follows.
    Explicit,
    /// Omit the annotation and infer it from the initializer.
    Infer,
}

/// An expression-form action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpressionAction {
    /// A symbol reference.
    Symbol,
    /// A scalar literal.
    Literal,
    /// A binary expression.
    Binary,
    /// A direct call.
    DirectCall,
    /// A conditional expression.
    If,
}

/// A literal action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LiteralAction {
    /// An integer value.
    Integer(i128),
    /// A string value.
    String(String),
    /// A boolean value.
    Bool(bool),
}

/// A binary-operator action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOperatorAction {
    /// The selected operator.
    Operator(BinaryOperator),
}

/// A pointer action to a symbol identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SymbolReferenceAction {
    /// The referenced symbol.
    pub symbol: SymbolId,
}

/// A direct-call argument-list action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallArgumentAction {
    /// Emit another argument expression.
    Argument,
    /// Finish the argument list.
    End,
}

/// An `if` else-branch action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IfElseAction {
    /// Emit an `else` block.
    Else,
    /// Omit the `else` branch.
    NoElse,
}

/// A typed structural action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    /// Root context selection.
    Root(RootAction),
    /// Module item-list control.
    ItemList(ItemListAction),
    /// Symbol declaration allocation.
    Declaration(DeclarationAction),
    /// Function parameter-list control.
    ParameterList(ParameterListAction),
    /// Primitive type selection.
    Type(TypeAction),
    /// Block entry/end control.
    Block(BlockAction),
    /// Local type annotation selection.
    TypeAnnotation(TypeAnnotationAction),
    /// Expression form selection.
    Expression(ExpressionAction),
    /// Literal value selection.
    Literal(LiteralAction),
    /// Binary operator selection.
    BinaryOperator(BinaryOperatorAction),
    /// General symbol pointer.
    SymbolReference(SymbolReferenceAction),
    /// Direct-call function pointer.
    DirectCallTarget(SymbolReferenceAction),
    /// Direct-call argument-list control.
    CallArgument(CallArgumentAction),
    /// `if` else-branch control.
    IfElse(IfElseAction),
}

/// One required action in a structural trace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceStep {
    /// The action category requested by the controller.
    pub need: Need,
    /// The selected action, whose variant matches `need`.
    pub action: Action,
}

/// A completed IR tree that cannot be represented by a valid controller trace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceError {
    /// A symbol reference is not visible at this source position.
    UnknownSymbol {
        /// The unavailable symbol.
        symbol: SymbolId,
    },
    /// A direct call targeted a visible non-function symbol.
    InvalidCallTarget {
        /// The non-function target.
        symbol: SymbolId,
    },
    /// A declaration reuses an identity already allocated in this root.
    DuplicateSymbolId {
        /// The reused identity.
        symbol: SymbolId,
    },
    /// A block or parameter scope contains the same declaration twice.
    DuplicateDeclaration {
        /// The duplicate declaration.
        symbol: SymbolId,
    },
}

/// Produces a deterministic, validated structural action trace for `root`.
pub fn trace(root: &Root) -> Result<Vec<TraceStep>, TraceError> {
    let mut tracer = Tracer::new(root)?;
    tracer.root(root)?;
    Ok(tracer.steps)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SymbolKind {
    Function,
    Parameter,
    Local,
}

struct Tracer {
    steps: Vec<TraceStep>,
    functions: BTreeSet<SymbolId>,
    allocated: BTreeMap<SymbolId, SymbolKind>,
    scopes: Vec<BTreeMap<SymbolId, SymbolKind>>,
}

impl Tracer {
    fn new(root: &Root) -> Result<Self, TraceError> {
        let mut functions = BTreeSet::new();
        let mut allocated = BTreeMap::new();
        if let Root::Module { items } = root {
            for item in items {
                let Item::Function(function) = item;
                if !functions.insert(function.name) {
                    return Err(TraceError::DuplicateSymbolId {
                        symbol: function.name,
                    });
                }
                if allocated
                    .insert(function.name, SymbolKind::Function)
                    .is_some()
                {
                    return Err(TraceError::DuplicateSymbolId {
                        symbol: function.name,
                    });
                }
            }
        }
        Ok(Self {
            steps: Vec::new(),
            functions,
            allocated,
            scopes: Vec::new(),
        })
    }

    fn root(&mut self, root: &Root) -> Result<(), TraceError> {
        match root {
            Root::Module { items } => {
                self.push(Need::Root, Action::Root(RootAction::Module));
                for item in items {
                    let Item::Function(function) = item;
                    self.push(Need::ItemOrEnd, Action::ItemList(ItemListAction::Function));
                    self.push(
                        Need::Declaration,
                        Action::Declaration(DeclarationAction::Function(function.name)),
                    );
                }
                self.push(Need::ItemOrEnd, Action::ItemList(ItemListAction::End));
                for item in items {
                    let Item::Function(function) = item;
                    self.function_body(function)?;
                }
            }
            Root::BlockFragment(block) => {
                self.push(Need::Root, Action::Root(RootAction::BlockFragment));
                self.block(block)?;
            }
        }
        Ok(())
    }

    fn function_body(&mut self, function: &crate::model::ir::Function) -> Result<(), TraceError> {
        self.scopes.push(BTreeMap::new());
        for parameter in &function.parameters {
            self.push(
                Need::ParameterOrEnd,
                Action::ParameterList(ParameterListAction::Parameter),
            );
            self.declare(parameter.name, SymbolKind::Parameter)?;
            self.push(
                Need::Declaration,
                Action::Declaration(DeclarationAction::Parameter(parameter.name)),
            );
            self.primitive_type(parameter.ty);
        }
        self.push(
            Need::ParameterOrEnd,
            Action::ParameterList(ParameterListAction::End),
        );
        self.primitive_type(function.return_type);
        self.block(&function.body)?;
        self.scopes.pop();
        Ok(())
    }

    fn block(&mut self, block: &Block) -> Result<(), TraceError> {
        self.scopes.push(BTreeMap::new());
        for statement in &block.statements {
            match statement {
                Statement::Let { name, ty, value } => {
                    self.push(Need::BlockEntryOrEnd, Action::Block(BlockAction::Let));
                    self.check_declaration(*name)?;
                    self.push(
                        Need::Declaration,
                        Action::Declaration(DeclarationAction::Local(*name)),
                    );
                    match ty {
                        Some(ty) => {
                            self.push(
                                Need::TypeAnnotation,
                                Action::TypeAnnotation(TypeAnnotationAction::Explicit),
                            );
                            self.primitive_type(*ty);
                        }
                        None => self.push(
                            Need::TypeAnnotation,
                            Action::TypeAnnotation(TypeAnnotationAction::Infer),
                        ),
                    }
                    self.expression(value)?;
                    self.finish_declaration(*name, SymbolKind::Local);
                }
                Statement::Expression(expression) => {
                    self.push(
                        Need::BlockEntryOrEnd,
                        Action::Block(BlockAction::ExpressionStatement),
                    );
                    self.expression(expression)?;
                }
            }
        }
        match &block.tail {
            Some(tail) => {
                self.push(
                    Need::BlockEntryOrEnd,
                    Action::Block(BlockAction::EndAndYield),
                );
                self.expression(tail)?;
            }
            None => self.push(
                Need::BlockEntryOrEnd,
                Action::Block(BlockAction::EndAndDiscard),
            ),
        }
        self.scopes.pop();
        Ok(())
    }

    fn expression(&mut self, expression: &Expression) -> Result<(), TraceError> {
        match expression {
            Expression::Symbol(symbol) => {
                self.require_visible(*symbol)?;
                self.push(
                    Need::Expression,
                    Action::Expression(ExpressionAction::Symbol),
                );
                self.push(
                    Need::SymbolReference,
                    Action::SymbolReference(SymbolReferenceAction { symbol: *symbol }),
                );
            }
            Expression::Literal(literal) => {
                self.push(
                    Need::Expression,
                    Action::Expression(ExpressionAction::Literal),
                );
                self.push(Need::Literal, Action::Literal(literal_action(literal)));
            }
            Expression::Binary {
                left,
                operator,
                right,
            } => {
                self.push(
                    Need::Expression,
                    Action::Expression(ExpressionAction::Binary),
                );
                self.push(
                    Need::BinaryOperator,
                    Action::BinaryOperator(BinaryOperatorAction::Operator(*operator)),
                );
                self.expression(left)?;
                self.expression(right)?;
            }
            Expression::Call {
                function,
                arguments,
            } => {
                self.require_function(*function)?;
                self.push(
                    Need::Expression,
                    Action::Expression(ExpressionAction::DirectCall),
                );
                self.push(
                    Need::DirectCallTarget,
                    Action::DirectCallTarget(SymbolReferenceAction { symbol: *function }),
                );
                for argument in arguments {
                    self.push(
                        Need::CallArgumentOrEnd,
                        Action::CallArgument(CallArgumentAction::Argument),
                    );
                    self.expression(argument)?;
                }
                self.push(
                    Need::CallArgumentOrEnd,
                    Action::CallArgument(CallArgumentAction::End),
                );
            }
            Expression::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.push(Need::Expression, Action::Expression(ExpressionAction::If));
                self.expression(condition)?;
                self.block(then_branch)?;
                match else_branch {
                    Some(else_branch) => {
                        self.push(Need::IfElse, Action::IfElse(IfElseAction::Else));
                        self.block(else_branch)?;
                    }
                    None => self.push(Need::IfElse, Action::IfElse(IfElseAction::NoElse)),
                }
            }
        }
        Ok(())
    }

    fn primitive_type(&mut self, ty: PrimitiveType) {
        self.push(Need::Type, Action::Type(TypeAction::Primitive(ty)));
    }

    fn declare(&mut self, symbol: SymbolId, kind: SymbolKind) -> Result<(), TraceError> {
        self.check_declaration(symbol)?;
        self.finish_declaration(symbol, kind);
        Ok(())
    }

    fn check_declaration(&self, symbol: SymbolId) -> Result<(), TraceError> {
        if self
            .scopes
            .last()
            .is_some_and(|scope| scope.contains_key(&symbol))
        {
            return Err(TraceError::DuplicateDeclaration { symbol });
        }
        if self.allocated.contains_key(&symbol) {
            return Err(TraceError::DuplicateSymbolId { symbol });
        }
        Ok(())
    }

    fn finish_declaration(&mut self, symbol: SymbolId, kind: SymbolKind) {
        self.allocated.insert(symbol, kind);
        self.scopes
            .last_mut()
            .expect("declarations require an active scope")
            .insert(symbol, kind);
    }

    fn require_visible(&self, symbol: SymbolId) -> Result<(), TraceError> {
        if self.functions.contains(&symbol)
            || self
                .scopes
                .iter()
                .rev()
                .any(|scope| scope.contains_key(&symbol))
        {
            Ok(())
        } else {
            Err(TraceError::UnknownSymbol { symbol })
        }
    }

    fn require_function(&self, symbol: SymbolId) -> Result<(), TraceError> {
        if self.functions.contains(&symbol) {
            Ok(())
        } else if self
            .scopes
            .iter()
            .rev()
            .any(|scope| scope.contains_key(&symbol))
        {
            Err(TraceError::InvalidCallTarget { symbol })
        } else {
            Err(TraceError::UnknownSymbol { symbol })
        }
    }

    fn push(&mut self, need: Need, action: Action) {
        self.steps.push(TraceStep { need, action });
    }
}

fn literal_action(literal: &Literal) -> LiteralAction {
    match literal {
        Literal::Integer(value) => LiteralAction::Integer(*value),
        Literal::String(value) => LiteralAction::String(value.clone()),
        Literal::Bool(value) => LiteralAction::Bool(*value),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Action, BinaryOperatorAction, BlockAction, CallArgumentAction, DeclarationAction,
        ExpressionAction, ItemListAction, Need, ParameterListAction, RootAction,
        SymbolReferenceAction, TraceError, TypeAction, trace,
    };
    use crate::model::ir::{
        BinaryOperator, Block, Expression, Function, Item, Literal, Parameter, PrimitiveType, Root,
        Statement, SymbolId,
    };

    #[test]
    fn traces_increment_deterministically() {
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

        assert_eq!(
            trace(&root).expect("valid trace"),
            vec![
                step(Need::Root, Action::Root(RootAction::Module)),
                step(Need::ItemOrEnd, Action::ItemList(ItemListAction::Function)),
                step(
                    Need::Declaration,
                    Action::Declaration(DeclarationAction::Function(SymbolId(1)))
                ),
                step(Need::ItemOrEnd, Action::ItemList(ItemListAction::End)),
                step(
                    Need::ParameterOrEnd,
                    Action::ParameterList(ParameterListAction::Parameter)
                ),
                step(
                    Need::Declaration,
                    Action::Declaration(DeclarationAction::Parameter(SymbolId(2)))
                ),
                step(
                    Need::Type,
                    Action::Type(TypeAction::Primitive(PrimitiveType::I32))
                ),
                step(
                    Need::ParameterOrEnd,
                    Action::ParameterList(ParameterListAction::End)
                ),
                step(
                    Need::Type,
                    Action::Type(TypeAction::Primitive(PrimitiveType::I32))
                ),
                step(
                    Need::BlockEntryOrEnd,
                    Action::Block(BlockAction::EndAndYield)
                ),
                step(
                    Need::Expression,
                    Action::Expression(ExpressionAction::Binary)
                ),
                step(
                    Need::BinaryOperator,
                    Action::BinaryOperator(BinaryOperatorAction::Operator(BinaryOperator::Add))
                ),
                step(
                    Need::Expression,
                    Action::Expression(ExpressionAction::Symbol)
                ),
                step(
                    Need::SymbolReference,
                    Action::SymbolReference(SymbolReferenceAction {
                        symbol: SymbolId(2)
                    })
                ),
                step(
                    Need::Expression,
                    Action::Expression(ExpressionAction::Literal)
                ),
                step(
                    Need::Literal,
                    Action::Literal(super::LiteralAction::Integer(1))
                ),
            ]
        );
    }

    #[test]
    fn traces_all_function_headers_before_bodies() {
        let root = Root::Module {
            items: vec![
                Item::Function(Function {
                    name: SymbolId(1),
                    parameters: vec![],
                    return_type: PrimitiveType::Unit,
                    body: Block {
                        statements: vec![],
                        tail: Some(Box::new(Expression::Symbol(SymbolId(2)))),
                    },
                }),
                Item::Function(Function {
                    name: SymbolId(2),
                    parameters: vec![],
                    return_type: PrimitiveType::Unit,
                    body: Block::default(),
                }),
            ],
        };

        let steps = trace(&root).expect("forward function reference is valid");
        assert_eq!(
            &steps[..6],
            &[
                step(Need::Root, Action::Root(RootAction::Module)),
                step(Need::ItemOrEnd, Action::ItemList(ItemListAction::Function)),
                step(
                    Need::Declaration,
                    Action::Declaration(DeclarationAction::Function(SymbolId(1)))
                ),
                step(Need::ItemOrEnd, Action::ItemList(ItemListAction::Function)),
                step(
                    Need::Declaration,
                    Action::Declaration(DeclarationAction::Function(SymbolId(2)))
                ),
                step(Need::ItemOrEnd, Action::ItemList(ItemListAction::End)),
            ]
        );
        assert_eq!(
            steps[6],
            step(
                Need::ParameterOrEnd,
                Action::ParameterList(ParameterListAction::End)
            )
        );
        assert_eq!(
            steps[10],
            step(
                Need::SymbolReference,
                Action::SymbolReference(SymbolReferenceAction {
                    symbol: SymbolId(2)
                })
            )
        );
        assert_eq!(
            steps[11],
            step(
                Need::ParameterOrEnd,
                Action::ParameterList(ParameterListAction::End)
            )
        );
    }

    #[test]
    fn distinguishes_block_tail_from_expression_statement() {
        let statement = Root::BlockFragment(Block {
            statements: vec![Statement::Expression(Expression::Literal(Literal::Bool(
                true,
            )))],
            tail: None,
        });
        let tail = Root::BlockFragment(Block {
            statements: vec![],
            tail: Some(Box::new(Expression::Literal(Literal::Bool(true)))),
        });

        let statement_end = trace(&statement)
            .expect("valid statement block")
            .last()
            .cloned();
        let tail_end = trace(&tail).expect("valid tail block")[1].clone();
        assert_eq!(
            statement_end,
            Some(step(
                Need::BlockEntryOrEnd,
                Action::Block(BlockAction::EndAndDiscard)
            ))
        );
        assert_eq!(
            tail_end,
            step(
                Need::BlockEntryOrEnd,
                Action::Block(BlockAction::EndAndYield)
            )
        );
    }

    #[test]
    fn rejects_invisible_symbols_and_local_call_targets() {
        let unknown = Root::BlockFragment(Block {
            statements: vec![],
            tail: Some(Box::new(Expression::Symbol(SymbolId(9)))),
        });
        assert_eq!(
            trace(&unknown),
            Err(TraceError::UnknownSymbol {
                symbol: SymbolId(9)
            })
        );

        let local_call = Root::BlockFragment(Block {
            statements: vec![Statement::Let {
                name: SymbolId(1),
                ty: None,
                value: Expression::Literal(Literal::Integer(0)),
            }],
            tail: Some(Box::new(Expression::Call {
                function: SymbolId(1),
                arguments: vec![],
            })),
        });
        assert_eq!(
            trace(&local_call),
            Err(TraceError::InvalidCallTarget {
                symbol: SymbolId(1)
            })
        );

        let duplicate_local = Root::BlockFragment(Block {
            statements: vec![
                Statement::Let {
                    name: SymbolId(2),
                    ty: None,
                    value: Expression::Literal(Literal::Integer(0)),
                },
                Statement::Let {
                    name: SymbolId(2),
                    ty: None,
                    value: Expression::Literal(Literal::Integer(1)),
                },
            ],
            tail: None,
        });
        assert_eq!(
            trace(&duplicate_local),
            Err(TraceError::DuplicateDeclaration {
                symbol: SymbolId(2)
            })
        );
    }

    #[test]
    fn rejects_duplicate_top_level_function_ids() {
        let function = || {
            Item::Function(Function {
                name: SymbolId(1),
                parameters: vec![],
                return_type: PrimitiveType::Unit,
                body: Block::default(),
            })
        };
        let root = Root::Module {
            items: vec![function(), function()],
        };

        assert_eq!(
            trace(&root),
            Err(TraceError::DuplicateSymbolId {
                symbol: SymbolId(1)
            })
        );
    }

    #[test]
    fn traces_parameter_and_call_list_continuations() {
        let root = Root::Module {
            items: vec![Item::Function(Function {
                name: SymbolId(1),
                parameters: vec![
                    Parameter {
                        name: SymbolId(2),
                        ty: PrimitiveType::I32,
                    },
                    Parameter {
                        name: SymbolId(3),
                        ty: PrimitiveType::I32,
                    },
                ],
                return_type: PrimitiveType::Unit,
                body: Block {
                    statements: vec![],
                    tail: Some(Box::new(Expression::Call {
                        function: SymbolId(1),
                        arguments: vec![
                            Expression::Symbol(SymbolId(2)),
                            Expression::Symbol(SymbolId(3)),
                        ],
                    })),
                },
            })],
        };
        let steps = trace(&root).expect("valid trace");
        assert_eq!(
            steps
                .iter()
                .filter(|step| step.need == Need::ParameterOrEnd)
                .count(),
            3
        );
        assert_eq!(
            steps
                .iter()
                .filter(|step| step.need == Need::CallArgumentOrEnd)
                .count(),
            3
        );
        assert!(steps.contains(&step(
            Need::CallArgumentOrEnd,
            Action::CallArgument(CallArgumentAction::End)
        )));
    }

    #[test]
    fn traverses_nested_if_and_calls() {
        let root = Root::Module {
            items: vec![Item::Function(Function {
                name: SymbolId(1),
                parameters: vec![],
                return_type: PrimitiveType::I32,
                body: Block {
                    statements: vec![],
                    tail: Some(Box::new(Expression::If {
                        condition: Box::new(Expression::Literal(Literal::Bool(true))),
                        then_branch: Block {
                            statements: vec![],
                            tail: Some(Box::new(Expression::Call {
                                function: SymbolId(1),
                                arguments: vec![],
                            })),
                        },
                        else_branch: Some(Block {
                            statements: vec![],
                            tail: Some(Box::new(Expression::If {
                                condition: Box::new(Expression::Literal(Literal::Bool(false))),
                                then_branch: Block::default(),
                                else_branch: None,
                            })),
                        }),
                    })),
                },
            })],
        };

        let steps = trace(&root).expect("valid nested trace");
        assert_eq!(
            steps
                .iter()
                .filter(|step| {
                    step.need == Need::Expression
                        && step.action == Action::Expression(ExpressionAction::If)
                })
                .count(),
            2
        );
        assert!(steps.contains(&step(
            Need::CallArgumentOrEnd,
            Action::CallArgument(CallArgumentAction::End)
        )));
    }

    fn step(need: Need, action: Action) -> super::TraceStep {
        super::TraceStep { need, action }
    }
}
