//! Stable categorical IDs and scalar features used at the model-input boundary.
//!
//! These numbers and scalar feature positions are a versioned serialization
//! contract. Any incompatible ID, count, feature-order, or scaling change must
//! increment [`EMBEDDING_SCHEMA_VERSION`] and regenerate encoded data.
//!
//! Existing IDs, including BOS, must not move within a schema version. New
//! action forms should receive unused global IDs after the existing range
//! rather than shifting later category offsets.

use crate::controller::{Action, DeclarationAction, LiteralAction};
use crate::training::{DeclarationKind, FixedTarget, LiteralKind, OutputHead};

use super::sequence::PreviousAction;

/// Version of the global embedding-ID and scalar-feature contract.
pub const EMBEDDING_SCHEMA_VERSION: u16 = 1;

pub const OUTPUT_HEAD_COUNT: u32 = 14;

pub const ENGLISH_MODALITY_ID: u32 = 0;
pub const STRUCTURAL_MODALITY_ID: u32 = 1;
pub const MODALITY_COUNT: u32 = 2;

/// Global previous-action vocabulary size, including BOS.
pub const PREVIOUS_ACTION_COUNT: u32 = 53;
/// Stable BOS ID. It remains fixed even if future forms are appended.
pub const PREVIOUS_ACTION_BOS_ID: u32 = 52;

pub const fn output_head_id(head: OutputHead) -> u32 {
    match head {
        OutputHead::Root => 0,
        OutputHead::ItemList => 1,
        OutputHead::DeclarationKind => 2,
        OutputHead::ParameterList => 3,
        OutputHead::Type => 4,
        OutputHead::Block => 5,
        OutputHead::TypeAnnotation => 6,
        OutputHead::Expression => 7,
        OutputHead::LiteralKind => 8,
        OutputHead::BinaryOperator => 9,
        OutputHead::SymbolPointer => 10,
        OutputHead::DirectCallTarget => 11,
        OutputHead::CallArgument => 12,
        OutputHead::IfElse => 13,
    }
}

/// Returns the stable global form ID for a previous structural action.
///
/// Pointer identities and literal payloads deliberately do not participate in
/// this ID. Their information is supplied through pointer context or payload
/// features instead.
pub fn previous_action_id(previous: &PreviousAction) -> u32 {
    let PreviousAction::Action(action) = previous else {
        return PREVIOUS_ACTION_BOS_ID;
    };

    match action {
        Action::Root(value) => FixedTarget::Root(*value).candidate_id() as u32,
        Action::ItemList(value) => 2 + FixedTarget::ItemList(*value).candidate_id() as u32,
        Action::Declaration(value) => {
            4 + match value {
                DeclarationAction::Function(_) => DeclarationKind::Function.candidate_id(),
                DeclarationAction::Parameter(_) => DeclarationKind::Parameter.candidate_id(),
                DeclarationAction::Local(_) => DeclarationKind::Local.candidate_id(),
            } as u32
        }
        Action::ParameterList(value) => {
            7 + FixedTarget::ParameterList(*value).candidate_id() as u32
        }
        Action::Type(value) => 9 + FixedTarget::Type(*value).candidate_id() as u32,
        Action::Block(value) => 20 + FixedTarget::Block(*value).candidate_id() as u32,
        Action::TypeAnnotation(value) => {
            24 + FixedTarget::TypeAnnotation(*value).candidate_id() as u32
        }
        Action::Expression(value) => 26 + FixedTarget::Expression(*value).candidate_id() as u32,
        Action::Literal(value) => {
            31 + match value {
                LiteralAction::Integer(_) => LiteralKind::Integer.candidate_id(),
                LiteralAction::String(_) => LiteralKind::String.candidate_id(),
                LiteralAction::Bool(_) => LiteralKind::Bool.candidate_id(),
            } as u32
        }
        Action::BinaryOperator(value) => {
            34 + FixedTarget::BinaryOperator(*value).candidate_id() as u32
        }
        Action::SymbolReference(_) => 46,
        Action::DirectCallTarget(_) => 47,
        Action::CallArgument(value) => 48 + FixedTarget::CallArgument(*value).candidate_id() as u32,
        Action::IfElse(value) => 50 + FixedTarget::IfElse(*value).candidate_id() as u32,
    }
}

/// Numeric features: sign, normalized magnitude bit length, four little-endian
/// magnitude limbs, zero flag, and small (`-16..=16`) flag.
pub fn integer_features(value: i128) -> [f32; 8] {
    let magnitude = value.unsigned_abs();
    let sign = if value < 0 {
        -1.0
    } else if value > 0 {
        1.0
    } else {
        0.0
    };
    let bit_length = (u128::BITS - magnitude.leading_zeros()) as f32 / u128::BITS as f32;
    let limb_scale = u32::MAX as f32;

    [
        sign,
        bit_length,
        (magnitude as u32) as f32 / limb_scale,
        ((magnitude >> 32) as u32) as f32 / limb_scale,
        ((magnitude >> 64) as u32) as f32 / limb_scale,
        ((magnitude >> 96) as u32) as f32 / limb_scale,
        f32::from(value == 0),
        f32::from((-16..=16).contains(&value)),
    ]
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiteralPayloadFeatures {
    pub is_integer: f32,
    pub integer: [f32; 8],
    pub is_bool: f32,
    pub bool_value: f32,
    pub is_string: f32,
    /// Log-scaled UTF-8 byte length, clipped at 65,535 bytes.
    pub string_byte_length: f32,
}

pub fn literal_payload_features(literal: &LiteralAction) -> LiteralPayloadFeatures {
    let mut features = LiteralPayloadFeatures {
        is_integer: 0.0,
        integer: [0.0; 8],
        is_bool: 0.0,
        bool_value: 0.0,
        is_string: 0.0,
        string_byte_length: 0.0,
    };
    match literal {
        LiteralAction::Integer(value) => {
            features.is_integer = 1.0;
            features.integer = integer_features(*value);
        }
        LiteralAction::Bool(value) => {
            features.is_bool = 1.0;
            features.bool_value = f32::from(*value);
        }
        LiteralAction::String(value) => {
            features.is_string = 1.0;
            let clipped = value.len().min(u16::MAX as usize) as f32;
            features.string_byte_length = clipped.ln_1p() / (u16::MAX as f32).ln_1p();
        }
    }
    features
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::*;
    use crate::model::ir::{BinaryOperator, PrimitiveType, SymbolId};

    #[test]
    fn output_head_ids_are_pinned_and_contiguous() {
        let heads = [
            OutputHead::Root,
            OutputHead::ItemList,
            OutputHead::DeclarationKind,
            OutputHead::ParameterList,
            OutputHead::Type,
            OutputHead::Block,
            OutputHead::TypeAnnotation,
            OutputHead::Expression,
            OutputHead::LiteralKind,
            OutputHead::BinaryOperator,
            OutputHead::SymbolPointer,
            OutputHead::DirectCallTarget,
            OutputHead::CallArgument,
            OutputHead::IfElse,
        ];
        assert_eq!(
            heads.map(output_head_id),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        );
        assert_eq!(heads.len() as u32, OUTPUT_HEAD_COUNT);
    }

    fn id(action: Action) -> u32 {
        previous_action_id(&PreviousAction::Action(action))
    }

    #[test]
    fn every_action_form_has_its_pinned_global_id() {
        let mut actual = vec![
            id(Action::Root(RootAction::Module)),
            id(Action::Root(RootAction::BlockFragment)),
            id(Action::ItemList(ItemListAction::Function)),
            id(Action::ItemList(ItemListAction::End)),
            id(Action::Declaration(DeclarationAction::Function(SymbolId(
                7,
            )))),
            id(Action::Declaration(DeclarationAction::Parameter(SymbolId(
                7,
            )))),
            id(Action::Declaration(DeclarationAction::Local(SymbolId(7)))),
            id(Action::ParameterList(ParameterListAction::Parameter)),
            id(Action::ParameterList(ParameterListAction::End)),
        ];
        for primitive in [
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
        ] {
            actual.push(id(Action::Type(TypeAction::Primitive(primitive))));
        }
        actual.extend([
            id(Action::Block(BlockAction::Let)),
            id(Action::Block(BlockAction::ExpressionStatement)),
            id(Action::Block(BlockAction::EndAndYield)),
            id(Action::Block(BlockAction::EndAndDiscard)),
            id(Action::TypeAnnotation(TypeAnnotationAction::Explicit)),
            id(Action::TypeAnnotation(TypeAnnotationAction::Infer)),
            id(Action::Expression(ExpressionAction::Symbol)),
            id(Action::Expression(ExpressionAction::Literal)),
            id(Action::Expression(ExpressionAction::Binary)),
            id(Action::Expression(ExpressionAction::DirectCall)),
            id(Action::Expression(ExpressionAction::If)),
            id(Action::Literal(LiteralAction::Integer(99))),
            id(Action::Literal(LiteralAction::String("payload".into()))),
            id(Action::Literal(LiteralAction::Bool(true))),
        ]);
        for operator in [
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
        ] {
            actual.push(id(Action::BinaryOperator(BinaryOperatorAction::Operator(
                operator,
            ))));
        }
        actual.extend([
            id(Action::SymbolReference(SymbolReferenceAction {
                symbol: SymbolId(1),
            })),
            id(Action::DirectCallTarget(SymbolReferenceAction {
                symbol: SymbolId(2),
            })),
            id(Action::CallArgument(CallArgumentAction::Argument)),
            id(Action::CallArgument(CallArgumentAction::End)),
            id(Action::IfElse(IfElseAction::Else)),
            id(Action::IfElse(IfElseAction::NoElse)),
        ]);

        assert_eq!(actual, (0..PREVIOUS_ACTION_BOS_ID).collect::<Vec<_>>());
        assert_eq!(
            previous_action_id(&PreviousAction::Bos),
            PREVIOUS_ACTION_BOS_ID
        );
        assert_eq!(PREVIOUS_ACTION_COUNT, 53);
    }

    #[test]
    fn identities_and_payloads_do_not_change_form_ids() {
        assert_eq!(
            id(Action::Declaration(DeclarationAction::Local(SymbolId(1)))),
            id(Action::Declaration(DeclarationAction::Local(SymbolId(999))))
        );
        assert_eq!(
            id(Action::Literal(LiteralAction::Integer(i128::MIN))),
            id(Action::Literal(LiteralAction::Integer(i128::MAX)))
        );
        assert_eq!(
            id(Action::Literal(LiteralAction::String("a".into()))),
            id(Action::Literal(LiteralAction::String("different".into())))
        );
        assert_eq!(
            id(Action::Literal(LiteralAction::Bool(false))),
            id(Action::Literal(LiteralAction::Bool(true)))
        );
    }

    #[test]
    fn integer_features_cover_extremes_and_flags() {
        for value in [i128::MIN, i128::MAX, -16, 0, 16, 17] {
            let first = integer_features(value);
            assert_eq!(first, integer_features(value));
            assert!(first.into_iter().all(f32::is_finite));
        }
        assert_eq!(integer_features(0)[6..], [1.0, 1.0]);
        assert_eq!(integer_features(-16)[7], 1.0);
        assert_eq!(integer_features(17)[7], 0.0);
        assert_eq!(integer_features(i128::MIN)[0], -1.0);
        assert_eq!(integer_features(i128::MAX)[0], 1.0);
    }

    #[test]
    fn literal_payload_features_are_non_hashing_metadata() {
        let false_features = literal_payload_features(&LiteralAction::Bool(false));
        let true_features = literal_payload_features(&LiteralAction::Bool(true));
        assert_ne!(false_features.bool_value, true_features.bool_value);

        let left = literal_payload_features(&LiteralAction::String("abc".into()));
        let right = literal_payload_features(&LiteralAction::String("xyz".into()));
        assert_eq!(left, right);
        assert_ne!(
            left,
            literal_payload_features(&LiteralAction::String("longer".into()))
        );
    }
}
