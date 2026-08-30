

struct Collection {
    symbol: SymbolId,
    items: Option<Vec<Field>>,
    isFixedLength: bool,
}

struct Function {
    symbol: SymbolId,
    generics: Option<Vec<GenericParam>>,
    returnType: Option<Type>,
    params: Option<Vec<Param>>,
    body: Option<Block>,
}

struct Block {
    returnType: Option<Type>,
    statements: Option<Vec<Statement>>,
    tail: Option<Expression>,
}

enum Statement {
    Block,
    FunctionDeclaration,
    Call,
    EnumDeclaration,
    StructDeclaration,
    Use,
    Mod,
    Return,
    If,         // Else can only happen after if...
    Match,      // Arms can only happen after match...
    Loop,
    While,
    For,
    Break,
    Continue,
    Let,
}

enum Type { // These are the types in the stdlib
    bool,
    char,
    f32,
    f64,
    fn,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    pointer,
    reference,
    slice,
    str,
    tuple,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    unit,
    f16,        // experimental
    f128,       // experimental
    never,      // experiemtnal
    custom(SymbolId),
}
