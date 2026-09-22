//! Symbol naming, independently of IR structure and rendering.
//!
//! IR construction is eager: symbols receive stable [`SymbolId`]s as they are
//! built. Naming may be deferred until a later `Need::SymbolName` phase. Callers
//! register the complete symbol universe for the context in their chosen order,
//! including ambient symbols referenced by a block fragment. The registry keeps
//! spellings as metadata, never as symbol identity. A future language-facing
//! layer may accumulate English tokens and submit the resulting candidate here.
//! Tokenization and candidate generation deliberately remain outside this
//! module.

use std::{collections::BTreeMap, error::Error, fmt};

use crate::model::ir::SymbolId;

/// The MVP's intentionally conservative, ASCII-only Rust identifier policy.
///
/// This subset reserves contextual keywords such as `union`, even where Rust
/// accepts them in some identifier positions.
#[derive(Clone, Copy, Debug, Default)]
pub struct MvpIdentifierPolicy;

impl MvpIdentifierPolicy {
    /// Returns whether `spelling` is a supported normal or raw identifier.
    pub fn is_valid(self, spelling: &str) -> bool {
        let identifier = spelling.strip_prefix("r#").unwrap_or(spelling);
        let is_raw = identifier.len() != spelling.len();

        if identifier.is_empty()
            || identifier == "_"
            || (!is_raw && is_keyword(identifier))
            || (is_raw && matches!(identifier, "crate" | "self" | "super" | "Self"))
        {
            return false;
        }

        let mut characters = identifier.bytes();
        matches!(characters.next(), Some(b'_' | b'a'..=b'z' | b'A'..=b'Z'))
            && characters.all(
                |character| matches!(character, b'_' | b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9'),
            )
    }

    /// Returns the collision key, treating `foo` and `r#foo` as aliases.
    ///
    /// Callers should validate the spelling before using its canonical form.
    pub fn canonical(self, spelling: &str) -> &str {
        spelling.strip_prefix("r#").unwrap_or(spelling)
    }
}

/// A failure to register or assign a symbol name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NameError {
    /// A declaration was registered more than once.
    DuplicateRegistration { symbol: SymbolId },
    /// Registration was attempted after naming began.
    RegistrationClosed { symbol: SymbolId },
    /// Assignment or resolution was attempted before naming began.
    NamingNotBegun { symbol: SymbolId },
    /// An operation without a particular symbol was attempted before naming began.
    NamingPhaseNotBegun,
    /// The spelling is not accepted by [`MvpIdentifierPolicy`].
    InvalidIdentifier { symbol: SymbolId, spelling: String },
    /// The symbol has not been registered.
    UnknownSymbol { symbol: SymbolId },
    /// Naming is single-assignment; an assigned symbol cannot be renamed.
    AlreadyNamed {
        symbol: SymbolId,
        attempted_spelling: String,
        existing_spelling: String,
    },
    /// The spelling aliases a name (assigned or fallback) held by another symbol.
    SpellingCollision {
        symbol: SymbolId,
        spelling: String,
        existing_symbol: SymbolId,
        existing_spelling: String,
    },
}

impl fmt::Display for NameError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateRegistration { symbol } => {
                write!(output, "symbol {symbol:?} is already registered")
            }
            Self::RegistrationClosed { symbol } => {
                write!(
                    output,
                    "cannot register symbol {symbol:?} after naming began"
                )
            }
            Self::NamingNotBegun { symbol } => {
                write!(
                    output,
                    "cannot name or resolve symbol {symbol:?} before naming begins"
                )
            }
            Self::NamingPhaseNotBegun => write!(output, "naming has not begun"),
            Self::InvalidIdentifier { symbol, spelling } => {
                write!(
                    output,
                    "{spelling:?} is not a valid identifier for symbol {symbol:?}"
                )
            }
            Self::UnknownSymbol { symbol } => {
                write!(output, "symbol {symbol:?} is not registered")
            }
            Self::AlreadyNamed {
                symbol,
                attempted_spelling,
                existing_spelling,
            } => write!(
                output,
                "cannot rename symbol {symbol:?} from {existing_spelling:?} to {attempted_spelling:?}"
            ),
            Self::SpellingCollision {
                symbol,
                spelling,
                existing_symbol,
                existing_spelling,
            } => write!(
                output,
                "{spelling:?} for symbol {symbol:?} collides with {existing_spelling:?} for symbol {existing_symbol:?}"
            ),
        }
    }
}

impl Error for NameError {}

#[derive(Clone, Debug)]
struct Entry {
    fallback: String,
    assigned: Option<String>,
}

/// Symbol spellings in caller-selected registration order.
///
/// The registered universe includes declarations and any ambient symbols that
/// may be referenced by the rendered context. Registration is explicit so its
/// order does not depend on numeric `SymbolId` ordering. Every registered symbol
/// immediately reserves its deterministic fallback, preventing later
/// assignments from colliding with names that have not yet been requested.
#[derive(Clone, Debug, Default)]
pub struct NameRegistry {
    order: Vec<SymbolId>,
    entries: BTreeMap<SymbolId, Entry>,
    spellings: BTreeMap<String, SymbolId>,
    sealed: bool,
}

impl NameRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a symbol in naming order during the registration phase.
    pub fn register(&mut self, symbol: SymbolId) -> Result<(), NameError> {
        if self.sealed {
            return Err(NameError::RegistrationClosed { symbol });
        }
        if self.entries.contains_key(&symbol) {
            return Err(NameError::DuplicateRegistration { symbol });
        }
        let fallback = fallback_name(symbol);
        self.spellings.insert(fallback.clone(), symbol);
        self.entries.insert(
            symbol,
            Entry {
                fallback,
                assigned: None,
            },
        );
        self.order.push(symbol);
        Ok(())
    }

    /// Ends registration and begins assignment and resolution.
    pub fn begin_naming(&mut self) {
        self.sealed = true;
    }

    /// Alias for [`Self::begin_naming`].
    pub fn seal(&mut self) {
        self.begin_naming();
    }

    /// Verifies that registration is complete and naming has begun.
    pub fn ensure_ready(&self) -> Result<(), NameError> {
        if self.sealed {
            Ok(())
        } else {
            Err(NameError::NamingPhaseNotBegun)
        }
    }

    /// Assigns a validated spelling to a registered, unnamed symbol.
    pub fn assign(
        &mut self,
        symbol: SymbolId,
        spelling: impl Into<String>,
    ) -> Result<(), NameError> {
        let spelling = spelling.into();
        if !self.sealed {
            return Err(NameError::NamingNotBegun { symbol });
        }
        let entry = self
            .entries
            .get(&symbol)
            .ok_or(NameError::UnknownSymbol { symbol })?;
        if let Some(existing_spelling) = &entry.assigned {
            return Err(NameError::AlreadyNamed {
                symbol,
                attempted_spelling: spelling,
                existing_spelling: existing_spelling.clone(),
            });
        }
        if !MvpIdentifierPolicy.is_valid(&spelling) {
            return Err(NameError::InvalidIdentifier { symbol, spelling });
        }

        let canonical = MvpIdentifierPolicy.canonical(&spelling);
        if let Some(&existing) = self.spellings.get(canonical)
            && existing != symbol
        {
            return Err(NameError::SpellingCollision {
                symbol,
                spelling,
                existing_symbol: existing,
                existing_spelling: self.entries[&existing]
                    .assigned
                    .clone()
                    .unwrap_or_else(|| self.entries[&existing].fallback.clone()),
            });
        }

        let entry = self.entries.get_mut(&symbol).expect("checked above");
        self.spellings.remove(&entry.fallback);
        self.spellings.insert(canonical.to_owned(), symbol);
        entry.assigned = Some(spelling);
        Ok(())
    }

    /// Returns the assigned spelling, or the deterministic fallback.
    pub fn name(&self, symbol: SymbolId) -> Result<&str, NameError> {
        if !self.sealed {
            return Err(NameError::NamingNotBegun { symbol });
        }
        let entry = self
            .entries
            .get(&symbol)
            .ok_or(NameError::UnknownSymbol { symbol })?;
        Ok(entry.assigned.as_deref().unwrap_or(&entry.fallback))
    }

    /// Iterates resolved spellings in registration order.
    pub fn iter(&self) -> Result<impl Iterator<Item = (SymbolId, &str)>, NameError> {
        self.ensure_ready()?;
        Ok(self.order.iter().map(|&symbol| {
            let entry = &self.entries[&symbol];
            (symbol, entry.assigned.as_deref().unwrap_or(&entry.fallback))
        }))
    }
}

/// Returns the stable deterministic fallback for `symbol`.
pub fn fallback_name(symbol: SymbolId) -> String {
    format!("_symbol_{}", symbol.0)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifier_policy_and_public_helpers_are_stable() {
        let policy = MvpIdentifierPolicy;
        for valid in ["foo", "_foo9", "r#match", "r#async"] {
            assert!(policy.is_valid(valid), "{valid}");
        }
        for invalid in ["", "_", "9foo", "match", "r#", "r#self", "r#crate", "é"] {
            assert!(!policy.is_valid(invalid), "{invalid}");
        }
        assert_eq!(policy.canonical("foo"), policy.canonical("r#foo"));
        assert_eq!(fallback_name(SymbolId(42)), "_symbol_42");
    }

    #[test]
    fn enforces_phases_and_duplicate_registration_without_mutation() {
        let mut names = NameRegistry::new();
        assert!(matches!(names.iter(), Err(NameError::NamingPhaseNotBegun)));
        assert_eq!(
            names.assign(SymbolId(1), "one"),
            Err(NameError::NamingNotBegun {
                symbol: SymbolId(1)
            })
        );
        names.register(SymbolId(1)).unwrap();
        assert!(matches!(names.iter(), Err(NameError::NamingPhaseNotBegun)));
        assert_eq!(
            names.register(SymbolId(1)),
            Err(NameError::DuplicateRegistration {
                symbol: SymbolId(1)
            })
        );
        names.seal();
        assert_eq!(
            names.register(SymbolId(2)),
            Err(NameError::RegistrationClosed {
                symbol: SymbolId(2)
            })
        );
        assert_eq!(names.name(SymbolId(1)), Ok("_symbol_1"));
    }

    #[test]
    fn all_registered_first_supports_any_assignment_order_and_registration_iteration() {
        let mut names = NameRegistry::new();
        for symbol in [SymbolId(8), SymbolId(2), SymbolId(5)] {
            names.register(symbol).unwrap();
        }
        names.begin_naming();
        names.assign(SymbolId(5), "five").unwrap();
        names.assign(SymbolId(8), "eight").unwrap();
        assert_eq!(
            names.iter().unwrap().collect::<Vec<_>>(),
            vec![
                (SymbolId(8), "eight"),
                (SymbolId(2), "_symbol_2"),
                (SymbolId(5), "five")
            ]
        );
    }

    #[test]
    fn raw_aliases_and_fallbacks_collide_with_full_context() {
        let mut names = NameRegistry::new();
        names.register(SymbolId(1)).unwrap();
        names.register(SymbolId(2)).unwrap();
        names.seal();
        names.assign(SymbolId(1), "r#foo").unwrap();
        assert_eq!(
            names.assign(SymbolId(2), "foo"),
            Err(NameError::SpellingCollision {
                symbol: SymbolId(2),
                spelling: "foo".into(),
                existing_symbol: SymbolId(1),
                existing_spelling: "r#foo".into(),
            })
        );
        assert_eq!(
            names.assign(SymbolId(1), "r#_symbol_2"),
            Err(NameError::AlreadyNamed {
                symbol: SymbolId(1),
                attempted_spelling: "r#_symbol_2".into(),
                existing_spelling: "r#foo".into(),
            })
        );
        let mut fallbacks = NameRegistry::new();
        fallbacks.register(SymbolId(1)).unwrap();
        fallbacks.register(SymbolId(2)).unwrap();
        fallbacks.seal();
        assert_ne!(
            fallbacks.name(SymbolId(1)).unwrap(),
            fallbacks.name(SymbolId(2)).unwrap()
        );
        assert_eq!(
            fallbacks.assign(SymbolId(1), "r#_symbol_2"),
            Err(NameError::SpellingCollision {
                symbol: SymbolId(1),
                spelling: "r#_symbol_2".into(),
                existing_symbol: SymbolId(2),
                existing_spelling: "_symbol_2".into(),
            })
        );
        assert_eq!(fallbacks.name(SymbolId(1)), Ok("_symbol_1"));
    }

    #[test]
    fn failed_operations_preserve_state_and_errors_are_contextual() {
        let mut names = NameRegistry::new();
        names.register(SymbolId(1)).unwrap();
        names.seal();
        assert_eq!(
            names.assign(SymbolId(1), "not-valid"),
            Err(NameError::InvalidIdentifier {
                symbol: SymbolId(1),
                spelling: "not-valid".into(),
            })
        );
        names.assign(SymbolId(1), "name").unwrap();
        assert_eq!(
            names.assign(SymbolId(1), "other"),
            Err(NameError::AlreadyNamed {
                symbol: SymbolId(1),
                attempted_spelling: "other".into(),
                existing_spelling: "name".into(),
            })
        );
        assert_eq!(names.name(SymbolId(1)), Ok("name"));
        assert_eq!(
            names.name(SymbolId(9)),
            Err(NameError::UnknownSymbol {
                symbol: SymbolId(9)
            })
        );
        assert_eq!(
            NameError::UnknownSymbol {
                symbol: SymbolId(9)
            }
            .to_string(),
            "symbol SymbolId(9) is not registered"
        );
        assert_eq!(
            NameError::SpellingCollision {
                symbol: SymbolId(2),
                spelling: "r#foo".into(),
                existing_symbol: SymbolId(1),
                existing_spelling: "foo".into(),
            }
            .to_string(),
            "\"r#foo\" for symbol SymbolId(2) collides with \"foo\" for symbol SymbolId(1)"
        );
    }
}
