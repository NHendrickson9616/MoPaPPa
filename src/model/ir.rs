struct Collection {
    pub symbol: SymbolId,
    pub items: Vec<Field>,
    pub is_fixed_length: bool,
}

struct Function {
    pub symbol: SymbolId,
    pub generics: Vec<GenericParam>,
    pub return_type: Option<Type>,
    pub params: Vec<Param>,
    pub body: Block,
}

struct Block {
    pub statements: Vec<Statement>,
    pub tail: Option<Box<Expression>>,
}

enum Statement {
    Let(LetStatement),
    Item(Item),
    Expression(Expression),
}

enum Expression {
    Symbol(SymbolId),
    Literal(Literal),

    Binary(BinaryExpression),
    Call(CallExpression),
    Block(Block),
    If(IfExpression),
    Match(MatchExpression),
    Loop(LoopExpression),
    Closure(Closure),

    Return(Option<Box<Expression>>),
    Break(Option<Box<Expression>>),
    Continue,
}

pub struct BinaryExpression {
    pub operator: BinaryOperator,
    pub left: Box<Expression>,
    pub right: Box<Expression>,
}

pub enum BinaryOperator {
    Add,
    Sub,
    Mul,
    Div,
}

pub struct IfExpression {
    pub condition: Box<Expression>,
    pub then: Block,
    pub else_expression: Option<Box<Expression>>, //Controller will have to limit to just if or block
}

pub struct LetStatement {
    pub symbol: SymbolId,
    pub mutable: bool,
    pub type_annotation: Option<Type>,
    pub value: Option<Expression>,
}

enum Type {
    // These are the types in the stdlib
    Bool,
    Char,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
    Pointer,
    Reference,
    Slice,
    Str,
    Tuple,
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
    Unit,  // () type
    F16,   // experimental
    F128,  // experimental
    Never, // experiemtnal
    Custom(SymbolId),
}
