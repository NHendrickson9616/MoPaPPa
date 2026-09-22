//! Deterministic rendering of the MVP IR to Rust source.

use std::{error::Error, fmt};

use crate::model::ir::{
    BinaryOperator, Block, Expression, Function, Item, Literal, PrimitiveType, Root, Statement,
    SymbolId,
};
use crate::naming::{NameError, NameRegistry};

/// A failure to resolve a symbol name while rendering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RenderError {
    /// The naming registry could not resolve a symbol.
    Naming(NameError),
}

impl fmt::Display for RenderError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Naming(error) => write!(output, "cannot render symbol name: {error}"),
        }
    }
}

impl Error for RenderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Naming(error) => Some(error),
        }
    }
}

impl From<NameError> for RenderError {
    fn from(error: NameError) -> Self {
        Self::Naming(error)
    }
}

/// Renders an IR root using names from a sealed registry.
///
/// The registry is borrowed immutably: naming must be completed before
/// rendering, and every symbol reached in `root` must have been registered.
pub fn render(root: &Root, names: &NameRegistry) -> Result<String, RenderError> {
    names.ensure_ready()?;
    Renderer { names }.root(root)
}

struct Renderer<'a> {
    names: &'a NameRegistry,
}

impl Renderer<'_> {
    fn root(&self, root: &Root) -> Result<String, RenderError> {
        match root {
            Root::Module { items } => Ok(items
                .iter()
                .map(|item| self.item(item))
                .collect::<Result<Vec<_>, _>>()?
                .join("\n\n")),
            Root::BlockFragment(block) => self.block(block, 0),
        }
    }

    fn item(&self, item: &Item) -> Result<String, RenderError> {
        match item {
            Item::Function(function) => self.function(function),
        }
    }

    fn function(&self, function: &Function) -> Result<String, RenderError> {
        let parameters = function
            .parameters
            .iter()
            .map(|parameter| {
                Ok(format!(
                    "{}: {}",
                    self.name(parameter.name)?,
                    self.ty(parameter.ty)
                ))
            })
            .collect::<Result<Vec<_>, RenderError>>()?
            .join(", ");
        Ok(format!(
            "fn {}({parameters}) -> {} {}",
            self.name(function.name)?,
            self.ty(function.return_type),
            self.block(&function.body, 0)?
        ))
    }

    fn block(&self, block: &Block, indent: usize) -> Result<String, RenderError> {
        if block.statements.is_empty() && block.tail.is_none() {
            return Ok("{}".to_owned());
        }

        let child_indent = indent + 4;
        let padding = " ".repeat(child_indent);
        let mut lines = block
            .statements
            .iter()
            .map(|statement| {
                Ok(format!(
                    "{padding}{}",
                    self.statement(statement, child_indent)?
                ))
            })
            .collect::<Result<Vec<_>, RenderError>>()?;
        if let Some(tail) = &block.tail {
            lines.push(format!(
                "{padding}{}",
                self.expression(tail, 0, child_indent)?
            ));
        }
        Ok(format!(
            "{{\n{}\n{}}}",
            lines.join("\n"),
            " ".repeat(indent)
        ))
    }

    fn statement(&self, statement: &Statement, indent: usize) -> Result<String, RenderError> {
        match statement {
            Statement::Let { name, ty, value } => {
                let annotation = ty
                    .map(|ty| format!(": {}", self.ty(ty)))
                    .unwrap_or_default();
                Ok(format!(
                    "let {}{annotation} = {};",
                    self.name(*name)?,
                    self.expression(value, 0, indent)?
                ))
            }
            Statement::Expression(expression) => {
                Ok(format!("{};", self.expression(expression, 0, indent)?))
            }
        }
    }

    fn expression(
        &self,
        expression: &Expression,
        parent_precedence: u8,
        indent: usize,
    ) -> Result<String, RenderError> {
        Ok(match expression {
            Expression::Symbol(symbol) => self.name(*symbol)?.to_owned(),
            Expression::Literal(literal) => self.literal(literal),
            Expression::Binary {
                left,
                operator,
                right,
            } => {
                let precedence = operator.precedence();
                let rendered = format!(
                    "{} {} {}",
                    self.expression(left, precedence, indent)?,
                    operator.source(),
                    self.expression(right, precedence, indent)?
                );
                if precedence <= parent_precedence {
                    format!("({rendered})")
                } else {
                    rendered
                }
            }
            Expression::Call {
                function,
                arguments,
            } => {
                let arguments = arguments
                    .iter()
                    .map(|argument| self.expression(argument, 0, indent))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(", ");
                format!("{}({arguments})", self.name(*function)?)
            }
            Expression::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let mut rendered = format!(
                    "if {} {}",
                    self.expression(condition, 0, indent)?,
                    self.block(then_branch, indent)?
                );
                if let Some(else_branch) = else_branch {
                    rendered.push_str(" else ");
                    rendered.push_str(&self.block(else_branch, indent)?);
                }
                if parent_precedence > 0 {
                    format!("({rendered})")
                } else {
                    rendered
                }
            }
        })
    }

    fn literal(&self, literal: &Literal) -> String {
        match literal {
            Literal::Integer(value) => value.to_string(),
            Literal::String(value) => format!("{value:?}"),
            Literal::Bool(value) => value.to_string(),
        }
    }

    fn name(&self, symbol: SymbolId) -> Result<&str, RenderError> {
        self.names.name(symbol).map_err(RenderError::from)
    }

    fn ty(&self, ty: PrimitiveType) -> &'static str {
        match ty {
            PrimitiveType::Bool => "bool",
            PrimitiveType::Char => "char",
            PrimitiveType::I32 => "i32",
            PrimitiveType::I64 => "i64",
            PrimitiveType::Isize => "isize",
            PrimitiveType::U32 => "u32",
            PrimitiveType::U64 => "u64",
            PrimitiveType::Usize => "usize",
            PrimitiveType::F32 => "f32",
            PrimitiveType::F64 => "f64",
            PrimitiveType::Unit => "()",
        }
    }
}

impl BinaryOperator {
    fn source(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Subtract => "-",
            Self::Multiply => "*",
            Self::Divide => "/",
            Self::Equal => "==",
            Self::NotEqual => "!=",
            Self::LessThan => "<",
            Self::LessOrEqual => "<=",
            Self::GreaterThan => ">",
            Self::GreaterOrEqual => ">=",
            Self::And => "&&",
            Self::Or => "||",
        }
    }

    fn precedence(self) -> u8 {
        match self {
            Self::Or => 1,
            Self::And => 2,
            Self::Equal
            | Self::NotEqual
            | Self::LessThan
            | Self::LessOrEqual
            | Self::GreaterThan
            | Self::GreaterOrEqual => 3,
            Self::Add | Self::Subtract => 4,
            Self::Multiply | Self::Divide => 5,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::{RenderError, render};
    use crate::model::ir::{
        BinaryOperator, Block, Expression, Function, Item, Literal, Parameter, PrimitiveType, Root,
        Statement, SymbolId,
    };
    use crate::naming::{NameError, NameRegistry};

    fn names(entries: &[(u32, Option<&str>)]) -> NameRegistry {
        let mut names = NameRegistry::new();
        for (id, _) in entries {
            names.register(SymbolId(*id)).unwrap();
        }
        names.seal();
        for (id, spelling) in entries {
            if let Some(spelling) = spelling {
                names.assign(SymbolId(*id), *spelling).unwrap();
            }
        }
        names
    }

    fn source(root: &Root, names: &NameRegistry) -> String {
        render(root, names).expect("test names must be valid")
    }

    #[test]
    fn renders_a_function_module() {
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

        let source = source(&root, &names(&[(1, Some("increment")), (2, Some("count"))]));
        syn::parse_file(&source).expect("rendered module must be Rust source");
        assert_eq!(
            source,
            "fn increment(count: i32) -> i32 {\n    count + 1\n}"
        );
    }

    #[test]
    fn renders_a_parseable_block_fragment() {
        let root = Root::BlockFragment(Block {
            statements: vec![Statement::Let {
                name: SymbolId(1),
                ty: Some(PrimitiveType::I32),
                value: Expression::Literal(Literal::Integer(41)),
            }],
            tail: Some(Box::new(Expression::Binary {
                left: Box::new(Expression::Symbol(SymbolId(1))),
                operator: BinaryOperator::Add,
                right: Box::new(Expression::Literal(Literal::Integer(1))),
            })),
        });

        let source = source(&root, &names(&[(1, Some("answer"))]));
        syn::parse_str::<syn::Block>(&source).expect("rendered fragment must be a Rust block");
        assert_eq!(source, "{\n    let answer: i32 = 41;\n    answer + 1\n}");
    }

    #[test]
    fn distinguishes_expression_statements_from_block_tails() {
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

        assert_eq!(source(&statement, &names(&[])), "{\n    true;\n}");
        assert_eq!(source(&tail, &names(&[])), "{\n    true\n}");
    }

    #[test]
    fn renders_if_and_calls_to_registered_ambient_functions() {
        let root = Root::BlockFragment(Block {
            statements: vec![],
            tail: Some(Box::new(Expression::If {
                condition: Box::new(Expression::Literal(Literal::Bool(true))),
                then_branch: Block {
                    statements: vec![],
                    tail: Some(Box::new(Expression::Call {
                        function: SymbolId(2),
                        arguments: vec![Expression::Literal(Literal::String(
                            "hello\n\"world\"".to_owned(),
                        ))],
                    })),
                },
                else_branch: Some(Block {
                    statements: vec![],
                    tail: Some(Box::new(Expression::Literal(Literal::Integer(0)))),
                }),
            })),
        });

        // Block fragments may reference ambient symbols supplied by their context.
        let source = source(&root, &names(&[(2, Some("log"))]));
        syn::parse_str::<syn::Block>(&source).expect("rendered if must parse");
        assert_eq!(
            source,
            "{\n    if true {\n        log(\"hello\\n\\\"world\\\"\")\n    } else {\n        0\n    }\n}"
        );
    }

    #[test]
    fn uses_stable_fallback_names_and_preserves_binary_grouping() {
        let right_nested = Root::BlockFragment(Block {
            statements: vec![],
            tail: Some(Box::new(Expression::Binary {
                left: Box::new(Expression::Symbol(SymbolId(9))),
                operator: BinaryOperator::Subtract,
                right: Box::new(Expression::Binary {
                    left: Box::new(Expression::Symbol(SymbolId(10))),
                    operator: BinaryOperator::Subtract,
                    right: Box::new(Expression::Literal(Literal::Integer(1))),
                }),
            })),
        });

        assert_eq!(
            source(&right_nested, &names(&[(9, None), (10, None)])),
            "{\n    _symbol_9 - (_symbol_10 - 1)\n}"
        );

        let left_nested = Root::BlockFragment(Block {
            statements: vec![],
            tail: Some(Box::new(Expression::Binary {
                left: Box::new(Expression::Binary {
                    left: Box::new(Expression::Symbol(SymbolId(9))),
                    operator: BinaryOperator::Subtract,
                    right: Box::new(Expression::Symbol(SymbolId(10))),
                }),
                operator: BinaryOperator::Subtract,
                right: Box::new(Expression::Literal(Literal::Integer(1))),
            })),
        });

        assert_eq!(
            source(&left_nested, &names(&[(9, None), (10, None)])),
            "{\n    (_symbol_9 - _symbol_10) - 1\n}"
        );
    }

    #[test]
    fn parenthesizes_if_expressions_in_binary_operands() {
        let conditional = Expression::If {
            condition: Box::new(Expression::Literal(Literal::Bool(true))),
            then_branch: Block {
                statements: vec![],
                tail: Some(Box::new(Expression::Literal(Literal::Integer(1)))),
            },
            else_branch: Some(Block {
                statements: vec![],
                tail: Some(Box::new(Expression::Literal(Literal::Integer(2)))),
            }),
        };
        let left = Root::BlockFragment(Block {
            statements: vec![],
            tail: Some(Box::new(Expression::Binary {
                left: Box::new(conditional.clone()),
                operator: BinaryOperator::Add,
                right: Box::new(Expression::Literal(Literal::Integer(3))),
            })),
        });
        let right = Root::BlockFragment(Block {
            statements: vec![],
            tail: Some(Box::new(Expression::Binary {
                left: Box::new(Expression::Literal(Literal::Integer(3))),
                operator: BinaryOperator::Add,
                right: Box::new(conditional),
            })),
        });

        for root in [&left, &right] {
            let rendered = source(root, &names(&[]));
            syn::parse_str::<syn::Block>(&rendered)
                .expect("an if binary operand must be parenthesized and parse");
        }
        assert_eq!(
            source(&left, &names(&[])),
            "{\n    (if true {\n        1\n    } else {\n        2\n    }) + 3\n}"
        );
        assert_eq!(
            source(&right, &names(&[])),
            "{\n    3 + (if true {\n        1\n    } else {\n        2\n    })\n}"
        );
    }

    #[test]
    fn accepts_names_validated_by_the_registry_including_raw_identifiers() {
        let root = Root::BlockFragment(Block {
            statements: vec![],
            tail: Some(Box::new(Expression::Symbol(SymbolId(1)))),
        });
        let mut names = NameRegistry::new();
        names.register(SymbolId(1)).unwrap();
        names.seal();
        assert_eq!(
            names.assign(SymbolId(1), "not-valid"),
            Err(NameError::InvalidIdentifier {
                symbol: SymbolId(1),
                spelling: "not-valid".to_owned(),
            })
        );

        names.assign(SymbolId(1), "r#match").unwrap();
        assert_eq!(source(&root, &names), "{\n    r#match\n}");
    }

    #[test]
    fn collisions_are_rejected_during_registry_assignment() {
        let mut names = NameRegistry::new();
        names.register(SymbolId(1)).unwrap();
        names.register(SymbolId(2)).unwrap();
        names.seal();
        names.assign(SymbolId(1), "same").unwrap();
        assert_eq!(
            names.assign(SymbolId(2), "same"),
            Err(NameError::SpellingCollision {
                symbol: SymbolId(2),
                spelling: "same".to_owned(),
                existing_symbol: SymbolId(1),
                existing_spelling: "same".to_owned(),
            })
        );

        let mut fallback_collision = NameRegistry::new();
        fallback_collision.register(SymbolId(1)).unwrap();
        fallback_collision.register(SymbolId(2)).unwrap();
        fallback_collision.seal();
        assert_eq!(
            fallback_collision.assign(SymbolId(2), "r#_symbol_1"),
            Err(NameError::SpellingCollision {
                symbol: SymbolId(2),
                spelling: "r#_symbol_1".to_owned(),
                existing_symbol: SymbolId(1),
                existing_spelling: "_symbol_1".to_owned(),
            })
        );
    }

    #[test]
    fn reports_unknown_and_unsealed_registries_while_rendering() {
        let root = Root::BlockFragment(Block {
            statements: vec![],
            tail: Some(Box::new(Expression::Symbol(SymbolId(1)))),
        });
        let mut unknown = NameRegistry::new();
        unknown.seal();
        assert_eq!(
            render(&root, &unknown),
            Err(RenderError::Naming(NameError::UnknownSymbol {
                symbol: SymbolId(1)
            }))
        );

        let mut unsealed = NameRegistry::new();
        unsealed.register(SymbolId(1)).unwrap();
        let error = render(&root, &unsealed).unwrap_err();
        assert_eq!(error, RenderError::Naming(NameError::NamingPhaseNotBegun));
        assert_eq!(
            error.to_string(),
            "cannot render symbol name: naming has not begun"
        );
        assert!(error.source().is_some());
    }

    #[test]
    fn requires_a_sealed_registry_even_when_no_symbols_are_resolved() {
        let roots = [
            Root::Module { items: vec![] },
            Root::BlockFragment(Block::default()),
            Root::BlockFragment(Block {
                statements: vec![],
                tail: Some(Box::new(Expression::Literal(Literal::Integer(1)))),
            }),
        ];
        let unsealed = NameRegistry::new();

        for root in roots {
            assert_eq!(
                render(&root, &unsealed),
                Err(RenderError::Naming(NameError::NamingPhaseNotBegun))
            );
        }
    }
}
