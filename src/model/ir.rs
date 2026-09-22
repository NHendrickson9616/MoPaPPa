//! Public, source-oriented intermediate representation for the executable MVP.
//!
//! The IR deliberately models only source constructs that the MVP can render.
//! Names are represented by stable [`SymbolId`] values rather than strings so
//! construction and naming can remain separate.

/// A stable identity for a declared symbol.
///
/// Rendering resolves each `SymbolId` through the naming registry for its
/// context. A registered symbol without an assigned spelling uses its
/// deterministic fallback; a missing registry entry is an error. For a
/// [`Root::BlockFragment`], the context may register ambient symbols in addition
/// to symbols declared within the fragment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymbolId(pub u32);

/// The explicit context in which an IR tree is rendered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Root {
    /// A compilation-unit-like sequence of top-level items.
    Module { items: Vec<Item> },
    /// A syntactically complete block suitable where Rust expects a block expression.
    BlockFragment(Block),
}

/// A top-level item supported by the MVP.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Item {
    /// A free function.
    Function(Function),
}

/// A free function declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Function {
    /// The function's declared symbol.
    pub name: SymbolId,
    /// The function's parameters, in source order.
    pub parameters: Vec<Parameter>,
    /// The explicit return type.
    pub return_type: PrimitiveType,
    /// The function body.
    pub body: Block,
}

/// A typed function parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parameter {
    /// The parameter's declared symbol.
    pub name: SymbolId,
    /// The parameter type.
    pub ty: PrimitiveType,
}

/// Primitive types supported by the MVP.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimitiveType {
    /// `bool`
    Bool,
    /// `char`
    Char,
    /// `i32`
    I32,
    /// `i64`
    I64,
    /// `isize`
    Isize,
    /// `u32`
    U32,
    /// `u64`
    U64,
    /// `usize`
    Usize,
    /// `f32`
    F32,
    /// `f64`
    F64,
    /// `()`
    Unit,
}

/// A block with statements and an optional, semicolon-free tail expression.
///
/// `tail` is intentionally separate from `statements`: a tail expression
/// evaluates to the block value, while [`Statement::Expression`] evaluates and
/// discards its value.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Block {
    /// Statements evaluated in source order.
    pub statements: Vec<Statement>,
    /// The block value, when present.
    pub tail: Option<Box<Expression>>,
}

/// A statement supported in a block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Statement {
    /// A local binding with an optional explicit primitive type.
    Let {
        /// The local symbol declared by this statement.
        name: SymbolId,
        /// An optional source-level type annotation.
        ty: Option<PrimitiveType>,
        /// The initializing expression.
        value: Expression,
    },
    /// An expression evaluated for side effects and terminated with `;`.
    Expression(Expression),
}

/// An expression supported by the MVP.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expression {
    /// A reference to a symbol in scope.
    Symbol(SymbolId),
    /// A scalar literal.
    Literal(Literal),
    /// An infix binary operation.
    Binary {
        /// The left operand.
        left: Box<Expression>,
        /// The operator.
        operator: BinaryOperator,
        /// The right operand.
        right: Box<Expression>,
    },
    /// A direct call to an internally identified function.
    Call {
        /// The function symbol; paths and expression-valued callees are out of scope.
        function: SymbolId,
        /// Call arguments, in source order.
        arguments: Vec<Expression>,
    },
    /// A conditional expression.
    If {
        /// The condition expression.
        condition: Box<Expression>,
        /// The block evaluated when the condition is true.
        then_branch: Block,
        /// The optional block evaluated when the condition is false.
        else_branch: Option<Block>,
    },
}

/// A scalar literal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Literal {
    /// An integer literal.
    Integer(i128),
    /// A UTF-8 string literal.
    String(String),
    /// A boolean literal.
    Bool(bool),
}

/// Binary operators supported by the MVP.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOperator {
    /// `+`
    Add,
    /// `-`
    Subtract,
    /// `*`
    Multiply,
    /// `/`
    Divide,
    /// `==`
    Equal,
    /// `!=`
    NotEqual,
    /// `<`
    LessThan,
    /// `<=`
    LessOrEqual,
    /// `>`
    GreaterThan,
    /// `>=`
    GreaterOrEqual,
    /// `&&`
    And,
    /// `||`
    Or,
}
