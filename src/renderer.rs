//! Deterministic rendering of the MVP IR to Rust source.

use std::collections::{BTreeMap, BTreeSet};

use crate::model::ir::{
    BinaryOperator, Block, Expression, Function, Item, Literal, PrimitiveType, Root, Statement,
    SymbolId,
};

/// Symbol spellings supplied to the renderer.
///
/// Values must be ASCII Rust identifiers or raw identifiers. Missing symbols are
/// rendered as deterministic valid identifiers of the form `_symbol_<number>`.
pub type NameMap = BTreeMap<SymbolId, String>;

/// An invalid caller-provided symbol spelling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RenderError {
    /// A name map entry is not a supported Rust identifier.
    InvalidSymbolName {
        /// The symbol associated with the invalid entry.
        symbol: SymbolId,
        /// The rejected name.
        name: String,
    },
    /// Two distinct symbols resolve to the same source identifier.
    ///
    /// Raw identifier spelling is ignored when detecting collisions.
    DuplicateSymbolName {
        /// The duplicate final spelling.
        name: String,
        /// The first symbol using this spelling.
        first: SymbolId,
        /// The other symbol using this spelling.
        second: SymbolId,
    },
}

/// Renders an IR root to deterministic Rust source.
///
/// Names for symbols used by `root` are validated before rendering, so
/// successful rendering cannot introduce invalid identifier syntax or aliases
/// through a caller-provided name. Unused name-map entries are ignored.
pub fn render(root: &Root, names: &NameMap) -> Result<String, RenderError> {
    validate_names(root, names)?;
    let mut renderer = Renderer { names };
    Ok(renderer.root(root))
}

fn validate_names(root: &Root, names: &NameMap) -> Result<(), RenderError> {
    let mut resolved_names = BTreeMap::new();
    for symbol in symbols_in(root) {
        let name = names
            .get(&symbol)
            .cloned()
            .unwrap_or_else(|| fallback_name(symbol));
        if !is_valid_identifier(&name) {
            return Err(RenderError::InvalidSymbolName { symbol, name });
        }
        if let Some(first) = resolved_names.insert(canonical_identifier(&name), symbol) {
            return Err(RenderError::DuplicateSymbolName {
                name,
                first,
                second: symbol,
            });
        }
    }
    Ok(())
}

fn canonical_identifier(name: &str) -> String {
    name.strip_prefix("r#").unwrap_or(name).to_owned()
}

fn symbols_in(root: &Root) -> BTreeSet<SymbolId> {
    let mut symbols = BTreeSet::new();
    match root {
        Root::Module { items } => {
            for item in items {
                let Item::Function(function) = item;
                symbols.insert(function.name);
                for parameter in &function.parameters {
                    symbols.insert(parameter.name);
                }
                collect_block_symbols(&function.body, &mut symbols);
            }
        }
        Root::BlockFragment(block) => collect_block_symbols(block, &mut symbols),
    }
    symbols
}

fn collect_block_symbols(block: &Block, symbols: &mut BTreeSet<SymbolId>) {
    for statement in &block.statements {
        match statement {
            Statement::Let { name, value, .. } => {
                symbols.insert(*name);
                collect_expression_symbols(value, symbols);
            }
            Statement::Expression(expression) => collect_expression_symbols(expression, symbols),
        }
    }
    if let Some(tail) = &block.tail {
        collect_expression_symbols(tail, symbols);
    }
}

fn collect_expression_symbols(expression: &Expression, symbols: &mut BTreeSet<SymbolId>) {
    match expression {
        Expression::Symbol(symbol) => {
            symbols.insert(*symbol);
        }
        Expression::Literal(_) => {}
        Expression::Binary { left, right, .. } => {
            collect_expression_symbols(left, symbols);
            collect_expression_symbols(right, symbols);
        }
        Expression::Call {
            function,
            arguments,
        } => {
            symbols.insert(*function);
            for argument in arguments {
                collect_expression_symbols(argument, symbols);
            }
        }
        Expression::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_expression_symbols(condition, symbols);
            collect_block_symbols(then_branch, symbols);
            if let Some(else_branch) = else_branch {
                collect_block_symbols(else_branch, symbols);
            }
        }
    }
}

fn fallback_name(symbol: SymbolId) -> String {
    format!("_symbol_{}", symbol.0)
}

fn is_valid_identifier(name: &str) -> bool {
    let identifier = name.strip_prefix("r#").unwrap_or(name);
    let is_raw = identifier.len() != name.len();

    if identifier.is_empty()
        || identifier == "_"
        || (!is_raw && is_keyword(identifier))
        || (is_raw && matches!(identifier, "crate" | "self" | "super" | "Self"))
    {
        return false;
    }

    let mut characters = identifier.bytes();
    matches!(characters.next(), Some(b'_' | b'a'..=b'z' | b'A'..=b'Z'))
        && characters
            .all(|character| matches!(character, b'_' | b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9'))
}

fn is_keyword(name: &str) -> bool {
    matches!(
        name,
        "as" | "break"
            | "const"
            | "continue"
            | "crate"
            | "else"
            | "enum"
            | "extern"
            | "false"
            | "fn"
            | "for"
            | "if"
            | "impl"
            | "in"
            | "let"
            | "loop"
            | "match"
            | "mod"
            | "move"
            | "mut"
            | "pub"
            | "ref"
            | "return"
            | "self"
            | "Self"
            | "static"
            | "struct"
            | "super"
            | "trait"
            | "true"
            | "type"
            | "unsafe"
            | "use"
            | "where"
            | "while"
            | "async"
            | "await"
            | "dyn"
            | "gen"
            | "abstract"
            | "become"
            | "box"
            | "do"
            | "final"
            | "macro"
            | "override"
            | "priv"
            | "try"
            | "typeof"
            | "unsized"
            | "virtual"
            | "yield"
            | "union"
    )
}

struct Renderer<'a> {
    names: &'a NameMap,
}

impl Renderer<'_> {
    fn root(&mut self, root: &Root) -> String {
        match root {
            Root::Module { items } => items
                .iter()
                .map(|item| self.item(item))
                .collect::<Vec<_>>()
                .join("\n\n"),
            Root::BlockFragment(block) => self.block(block, 0),
        }
    }

    fn item(&mut self, item: &Item) -> String {
        match item {
            Item::Function(function) => self.function(function),
        }
    }

    fn function(&mut self, function: &Function) -> String {
        let parameters = function
            .parameters
            .iter()
            .map(|parameter| format!("{}: {}", self.name(parameter.name), self.ty(parameter.ty)))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "fn {}({parameters}) -> {} {}",
            self.name(function.name),
            self.ty(function.return_type),
            self.block(&function.body, 0)
        )
    }

    fn block(&mut self, block: &Block, indent: usize) -> String {
        if block.statements.is_empty() && block.tail.is_none() {
            return "{}".to_owned();
        }

        let child_indent = indent + 4;
        let padding = " ".repeat(child_indent);
        let mut lines = block
            .statements
            .iter()
            .map(|statement| format!("{padding}{}", self.statement(statement, child_indent)))
            .collect::<Vec<_>>();
        if let Some(tail) = &block.tail {
            lines.push(format!(
                "{padding}{}",
                self.expression(tail, 0, child_indent)
            ));
        }
        format!("{{\n{}\n{}}}", lines.join("\n"), " ".repeat(indent))
    }

    fn statement(&mut self, statement: &Statement, indent: usize) -> String {
        match statement {
            Statement::Let { name, ty, value } => {
                let annotation = ty
                    .map(|ty| format!(": {}", self.ty(ty)))
                    .unwrap_or_default();
                format!(
                    "let {}{annotation} = {};",
                    self.name(*name),
                    self.expression(value, 0, indent)
                )
            }
            Statement::Expression(expression) => {
                format!("{};", self.expression(expression, 0, indent))
            }
        }
    }

    fn expression(
        &mut self,
        expression: &Expression,
        parent_precedence: u8,
        indent: usize,
    ) -> String {
        match expression {
            Expression::Symbol(symbol) => self.name(*symbol),
            Expression::Literal(literal) => self.literal(literal),
            Expression::Binary {
                left,
                operator,
                right,
            } => {
                let precedence = operator.precedence();
                let rendered = format!(
                    "{} {} {}",
                    self.expression(left, precedence, indent),
                    operator.source(),
                    self.expression(right, precedence, indent)
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
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}({arguments})", self.name(*function))
            }
            Expression::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let mut rendered = format!(
                    "if {} {}",
                    self.expression(condition, 0, indent),
                    self.block(then_branch, indent)
                );
                if let Some(else_branch) = else_branch {
                    rendered.push_str(" else ");
                    rendered.push_str(&self.block(else_branch, indent));
                }
                if parent_precedence > 0 {
                    format!("({rendered})")
                } else {
                    rendered
                }
            }
        }
    }

    fn literal(&self, literal: &Literal) -> String {
        match literal {
            Literal::Integer(value) => value.to_string(),
            Literal::String(value) => format!("{value:?}"),
            Literal::Bool(value) => value.to_string(),
        }
    }

    fn name(&self, symbol: SymbolId) -> String {
        self.names
            .get(&symbol)
            .cloned()
            .unwrap_or_else(|| fallback_name(symbol))
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
    use super::{NameMap, RenderError, render};
    use crate::model::ir::{
        BinaryOperator, Block, Expression, Function, Item, Literal, Parameter, PrimitiveType, Root,
        Statement, SymbolId,
    };

    fn names(entries: &[(u32, &str)]) -> NameMap {
        entries
            .iter()
            .map(|(id, name)| (SymbolId(*id), (*name).to_owned()))
            .collect()
    }

    fn source(root: &Root, names: &NameMap) -> String {
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

        let source = source(&root, &names(&[(1, "increment"), (2, "count")]));
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

        let source = source(&root, &names(&[(1, "answer")]));
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

        assert_eq!(source(&statement, &NameMap::new()), "{\n    true;\n}");
        assert_eq!(source(&tail, &NameMap::new()), "{\n    true\n}");
    }

    #[test]
    fn renders_if_and_direct_calls() {
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

        let source = source(&root, &names(&[(2, "log")]));
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
            source(&right_nested, &NameMap::new()),
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
            source(&left_nested, &NameMap::new()),
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
            let rendered = source(root, &NameMap::new());
            syn::parse_str::<syn::Block>(&rendered)
                .expect("an if binary operand must be parenthesized and parse");
        }
        assert_eq!(
            source(&left, &NameMap::new()),
            "{\n    (if true {\n        1\n    } else {\n        2\n    }) + 3\n}"
        );
        assert_eq!(
            source(&right, &NameMap::new()),
            "{\n    3 + (if true {\n        1\n    } else {\n        2\n    })\n}"
        );
    }

    #[test]
    fn validates_used_names_and_accepts_raw_identifiers() {
        let root = Root::BlockFragment(Block {
            statements: vec![],
            tail: Some(Box::new(Expression::Symbol(SymbolId(1)))),
        });
        assert_eq!(
            render(&root, &names(&[(1, "not-valid")])),
            Err(RenderError::InvalidSymbolName {
                symbol: SymbolId(1),
                name: "not-valid".to_owned(),
            })
        );

        assert_eq!(
            source(&root, &names(&[(1, "r#match")])),
            "{\n    r#match\n}"
        );
    }

    #[test]
    fn rejects_supplied_and_fallback_name_collisions() {
        let root = Root::BlockFragment(Block {
            statements: vec![],
            tail: Some(Box::new(Expression::Binary {
                left: Box::new(Expression::Symbol(SymbolId(1))),
                operator: BinaryOperator::Add,
                right: Box::new(Expression::Symbol(SymbolId(2))),
            })),
        });

        assert_eq!(
            render(&root, &names(&[(1, "same"), (2, "same")])),
            Err(RenderError::DuplicateSymbolName {
                name: "same".to_owned(),
                first: SymbolId(1),
                second: SymbolId(2),
            })
        );
        assert_eq!(
            render(&root, &names(&[(2, "_symbol_1")])),
            Err(RenderError::DuplicateSymbolName {
                name: "_symbol_1".to_owned(),
                first: SymbolId(1),
                second: SymbolId(2),
            })
        );
        assert_eq!(
            render(&root, &names(&[(1, "foo"), (2, "r#foo")])),
            Err(RenderError::DuplicateSymbolName {
                name: "r#foo".to_owned(),
                first: SymbolId(1),
                second: SymbolId(2),
            })
        );
        assert_eq!(
            render(&root, &names(&[(2, "r#_symbol_1")])),
            Err(RenderError::DuplicateSymbolName {
                name: "r#_symbol_1".to_owned(),
                first: SymbolId(1),
                second: SymbolId(2),
            })
        );
    }
}
