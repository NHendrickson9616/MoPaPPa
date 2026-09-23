//! Pure checkpoint metadata and validation; this module never loads tensors.
//! Dimension agreement is diagnostic only. A future loader must authorize a
//! mapping with [`validate_loadable`] before reading any tensor.

use super::decoder::DecoderConfig;
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fmt,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointMetadata {
    pub architecture: DecoderArchitecture,
    pub qkv_layout: QkvLayout,
    pub tokenizer: TokenizerFingerprint,
    pub provenance: CheckpointProvenance,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderArchitecture {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub d_ff: usize,
    pub max_sequence_length: usize,
    pub norm: NormSpec,
    pub norm_placement: NormPlacement,
    pub final_norm: bool,
    pub activation: Activation,
    pub mlp: MlpKind,
    pub position: PositionHandling,
    pub attention: AttentionKind,
    pub biases: BiasConvention,
    pub tied_embedding_lm_head: bool,
}
pub type TargetArchitecture = DecoderArchitecture;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NormSpec {
    pub kind: NormKind,
    pub epsilon_bits: u64,
}
impl NormSpec {
    pub fn new(kind: NormKind, epsilon: f64) -> Self {
        Self {
            kind,
            epsilon_bits: epsilon.to_bits(),
        }
    }
    pub fn epsilon(self) -> f64 {
        f64::from_bits(self.epsilon_bits)
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormKind {
    LayerNorm,
    RmsNorm,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormPlacement {
    Pre,
    Post,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Activation {
    Gelu,
    GeluApproximate,
    Relu,
    Silu,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlpKind {
    Ordinary,
    Gated,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionKind {
    MultiHead,
    GroupedQuery,
    MultiQuery,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PositionHandling {
    None,
    External,
    Learned { max_positions: usize },
    Rope(RopeSpec),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RopeSpec {
    pub rotary_dim: usize,
    pub interleaved: bool,
    pub theta_bits: u64,
    pub scaling: Option<LinearRopeScaling>,
    pub max_positions: usize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinearRopeScaling {
    pub factor_bits: u64,
    pub original_max_positions: usize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BiasConvention {
    pub attention_qkv: bool,
    pub attention_output: bool,
    pub feed_forward: bool,
    pub norms: bool,
    pub lm_head: bool,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QkvLayout {
    Separate,
    Fused,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoaderCapabilities {
    pub target_qkv_layout: QkvLayout,
    pub allow_qkv_repacking: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerFingerprint {
    pub identifier: String,
    pub revision: String,
    pub vocabulary_digest: String,
    pub config_digest: String,
    pub vocab_size: usize,
    pub special_token_ids: Vec<(String, u32)>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointProvenance {
    pub identifier: String,
    pub revision: String,
    pub digest: String,
    pub license: LicenseMetadata,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LicenseMetadata {
    Unknown,
    Spdx(String),
    Verbatim(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Mismatch {
    pub field: &'static str,
    pub expected: String,
    pub found: String,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompatibilityError {
    InvalidLocalConfig(String),
    InvalidMetadata(Vec<Mismatch>),
    DimensionMismatch(Vec<Mismatch>),
    NotLoadable(Vec<Mismatch>),
}
impl CompatibilityError {
    fn metadata(v: Vec<Mismatch>) -> Self {
        debug_assert!(!v.is_empty());
        Self::InvalidMetadata(v)
    }
    fn dimensions(v: Vec<Mismatch>) -> Self {
        debug_assert!(!v.is_empty());
        Self::DimensionMismatch(v)
    }
    fn load(v: Vec<Mismatch>) -> Self {
        debug_assert!(!v.is_empty());
        Self::NotLoadable(v)
    }
}
impl fmt::Display for CompatibilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (label, values) = match self {
            Self::InvalidLocalConfig(s) => return write!(f, "invalid local decoder config: {s}"),
            Self::InvalidMetadata(v) => ("invalid checkpoint metadata", v),
            Self::DimensionMismatch(v) => ("decoder dimensions differ", v),
            Self::NotLoadable(v) => ("checkpoint is not loadable", v),
        };
        write!(f, "{label}")?;
        for v in values {
            write!(
                f,
                "; {}: expected {}, found {}",
                v.field, v.expected, v.found
            )?;
        }
        Ok(())
    }
}
impl Error for CompatibilityError {}

/// Checks only dimensions shared with `DecoderConfig`; it does not authorize loading.
pub fn validate_decoder_dimensions(
    m: &CheckpointMetadata,
    c: &DecoderConfig,
) -> Result<(), CompatibilityError> {
    c.validate()
        .map_err(|e| CompatibilityError::InvalidLocalConfig(e.to_string()))?;
    validate_metadata(m)?;
    let a = &m.architecture;
    let mut v = Vec::new();
    check(&mut v, "vocab_size", c.vocab_size, a.vocab_size);
    check(&mut v, "d_model", c.d_model, a.d_model);
    check(&mut v, "num_layers", c.num_layers, a.num_layers);
    check(&mut v, "num_heads", c.num_heads, a.num_heads);
    check(&mut v, "head_dim", c.d_model / c.num_heads, a.head_dim);
    check(&mut v, "d_ff", c.d_ff, a.d_ff);
    check(
        &mut v,
        "max_sequence_length",
        c.max_seq_len,
        a.max_sequence_length,
    );
    check(
        &mut v,
        "norm.epsilon",
        c.norm_eps.to_bits(),
        a.norm.epsilon_bits,
    );
    if v.is_empty() {
        Ok(())
    } else {
        Err(CompatibilityError::dimensions(v))
    }
}

/// Requires exact logical architecture and tokenizer identity. QKV repacking is
/// the sole storage-layout transformation represented here.
pub fn validate_loadable(
    m: &CheckpointMetadata,
    target: &TargetArchitecture,
    tokenizer: &TokenizerFingerprint,
    capabilities: LoaderCapabilities,
) -> Result<(), CompatibilityError> {
    validate_metadata(m)?;
    let invalid_target = validate_architecture(target);
    if !invalid_target.is_empty() {
        return Err(CompatibilityError::metadata(invalid_target));
    }
    let invalid_tokenizer = validate_tokenizer(tokenizer, target.vocab_size);
    if !invalid_tokenizer.is_empty() {
        return Err(CompatibilityError::metadata(invalid_tokenizer));
    }
    let mut v = Vec::new();
    check(&mut v, "architecture", target, &m.architecture);
    check(&mut v, "tokenizer", tokenizer, &m.tokenizer);
    if m.qkv_layout != capabilities.target_qkv_layout && !capabilities.allow_qkv_repacking {
        v.push(mm(
            "qkv_layout",
            capabilities.target_qkv_layout,
            m.qkv_layout,
        ));
    }
    if v.is_empty() {
        Ok(())
    } else {
        Err(CompatibilityError::load(v))
    }
}

fn validate_metadata(m: &CheckpointMetadata) -> Result<(), CompatibilityError> {
    let mut v = validate_architecture(&m.architecture);
    v.extend(validate_tokenizer(&m.tokenizer, m.architecture.vocab_size));
    text(&mut v, "checkpoint.identifier", &m.provenance.identifier);
    text(&mut v, "checkpoint.revision", &m.provenance.revision);
    text(&mut v, "checkpoint.digest", &m.provenance.digest);
    match &m.provenance.license {
        LicenseMetadata::Spdx(s) | LicenseMetadata::Verbatim(s) => {
            text(&mut v, "checkpoint.license", s)
        }
        LicenseMetadata::Unknown => {}
    }
    if v.is_empty() {
        Ok(())
    } else {
        Err(CompatibilityError::metadata(v))
    }
}
fn validate_tokenizer(
    tokenizer: &TokenizerFingerprint,
    architecture_vocab_size: usize,
) -> Vec<Mismatch> {
    let mut v = Vec::new();
    text(&mut v, "tokenizer.identifier", &tokenizer.identifier);
    text(&mut v, "tokenizer.revision", &tokenizer.revision);
    text(
        &mut v,
        "tokenizer.vocabulary_digest",
        &tokenizer.vocabulary_digest,
    );
    text(&mut v, "tokenizer.config_digest", &tokenizer.config_digest);
    if tokenizer.vocab_size != architecture_vocab_size {
        v.push(mm(
            "tokenizer.vocab_size",
            architecture_vocab_size,
            tokenizer.vocab_size,
        ));
    }

    let mut names = HashSet::new();
    let mut ids = HashMap::new();
    for (name, id) in &tokenizer.special_token_ids {
        text(&mut v, "tokenizer.special_token_ids.name", name);
        if (*id as usize) >= tokenizer.vocab_size {
            v.push(mm(
                "tokenizer.special_token_ids.id",
                format!("< {}", tokenizer.vocab_size),
                id,
            ));
        }
        if !names.insert(name.as_str()) {
            v.push(mm(
                "tokenizer.special_token_ids.name",
                "unique special-token name",
                name,
            ));
        }
        match ids.insert(*id, name.as_str()) {
            Some(existing_name) if existing_name != name => {
                v.push(mm(
                    "tokenizer.special_token_ids.id",
                    "unique ID (aliases are not represented)",
                    id,
                ));
            }
            _ => {}
        }
    }
    v
}
fn validate_architecture(a: &DecoderArchitecture) -> Vec<Mismatch> {
    let mut v = Vec::new();
    for (name, n) in [
        ("vocab_size", a.vocab_size),
        ("d_model", a.d_model),
        ("num_layers", a.num_layers),
        ("num_heads", a.num_heads),
        ("num_kv_heads", a.num_kv_heads),
        ("head_dim", a.head_dim),
        ("d_ff", a.d_ff),
        ("max_sequence_length", a.max_sequence_length),
    ] {
        if n == 0 {
            v.push(mm(name, "nonzero", n));
        }
    }
    if a.num_heads > 0 && a.num_kv_heads > 0 && !a.num_heads.is_multiple_of(a.num_kv_heads) {
        v.push(mm("num_kv_heads", "divisor of num_heads", a.num_kv_heads));
    }
    if a.num_heads.checked_mul(a.head_dim) != Some(a.d_model) {
        v.push(mm(
            "heads_x_head_dim",
            a.d_model,
            a.num_heads.saturating_mul(a.head_dim),
        ));
    }
    positive(&mut v, "norm.epsilon", a.norm.epsilon());
    match a.position {
        PositionHandling::Learned { max_positions } if max_positions < a.max_sequence_length => v
            .push(mm(
                "position.max_positions",
                format!(">= {}", a.max_sequence_length),
                max_positions,
            )),
        PositionHandling::Rope(r) => {
            if r.rotary_dim == 0 || r.rotary_dim > a.head_dim || !r.rotary_dim.is_multiple_of(2) {
                v.push(mm(
                    "position.rope.rotary_dim",
                    format!("even, <= {}", a.head_dim),
                    r.rotary_dim,
                ));
            }
            positive(&mut v, "position.rope.theta", f64::from_bits(r.theta_bits));
            if r.max_positions < a.max_sequence_length {
                v.push(mm(
                    "position.rope.max_positions",
                    format!(">= {}", a.max_sequence_length),
                    r.max_positions,
                ));
            }
            if let Some(s) = r.scaling {
                positive(
                    &mut v,
                    "position.rope.scaling.factor",
                    f64::from_bits(s.factor_bits),
                );
                if s.original_max_positions == 0 {
                    v.push(mm(
                        "position.rope.scaling.original_max_positions",
                        "nonzero",
                        0,
                    ));
                }
            }
        }
        _ => {}
    }
    let attention = if a.num_kv_heads == a.num_heads {
        AttentionKind::MultiHead
    } else if a.num_kv_heads == 1 {
        AttentionKind::MultiQuery
    } else {
        AttentionKind::GroupedQuery
    };
    if a.attention != attention {
        v.push(mm("attention", attention, a.attention));
    }
    v
}
fn positive(v: &mut Vec<Mismatch>, field: &'static str, n: f64) {
    if !n.is_finite() || n <= 0.0 {
        v.push(mm(field, "finite and positive", n));
    }
}
fn text(v: &mut Vec<Mismatch>, field: &'static str, s: &str) {
    if s.trim().is_empty() {
        v.push(mm(field, "nonempty immutable value", s));
    }
}
fn check<T: fmt::Debug + PartialEq>(
    v: &mut Vec<Mismatch>,
    field: &'static str,
    expected: T,
    found: T,
) {
    if expected != found {
        v.push(mm(field, expected, found));
    }
}
fn mm(field: &'static str, expected: impl fmt::Debug, found: impl fmt::Debug) -> Mismatch {
    Mismatch {
        field,
        expected: format!("{expected:?}"),
        found: format!("{found:?}"),
    }
}

/// Canonical destinations make ownership impossible for a caller to relabel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParameterDestination {
    Backbone,
    TokenEmbedding,
    FreeTokenOutput(FreeTokenOutput),
    ActionEmbedding,
    NeedEmbedding,
    StateEmbedding,
    ModalityEmbedding,
    StructuralHead,
    PointerHead,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FreeTokenOutput {
    TiedTokenEmbeddingAlias,
    UntiedTensor,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorOwnership {
    Checkpoint,
    NewlyInitialized,
}
impl ParameterDestination {
    pub const fn ownership(self) -> TensorOwnership {
        match self {
            Self::Backbone | Self::TokenEmbedding | Self::FreeTokenOutput(_) => {
                TensorOwnership::Checkpoint
            }
            _ => TensorOwnership::NewlyInitialized,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn arch() -> DecoderArchitecture {
        DecoderArchitecture {
            vocab_size: 32,
            d_model: 16,
            num_layers: 3,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 4,
            d_ff: 64,
            max_sequence_length: 128,
            norm: NormSpec::new(NormKind::LayerNorm, 1e-5),
            norm_placement: NormPlacement::Pre,
            final_norm: true,
            activation: Activation::Gelu,
            mlp: MlpKind::Ordinary,
            position: PositionHandling::External,
            attention: AttentionKind::MultiHead,
            biases: BiasConvention {
                attention_qkv: true,
                attention_output: true,
                feed_forward: true,
                norms: true,
                lm_head: false,
            },
            tied_embedding_lm_head: false,
        }
    }
    fn tok() -> TokenizerFingerprint {
        TokenizerFingerprint {
            identifier: "tok".into(),
            revision: "rev".into(),
            vocabulary_digest: "v-digest".into(),
            config_digest: "c-digest".into(),
            vocab_size: 32,
            special_token_ids: vec![("eos".into(), 1)],
        }
    }
    fn meta() -> CheckpointMetadata {
        CheckpointMetadata {
            architecture: arch(),
            qkv_layout: QkvLayout::Separate,
            tokenizer: tok(),
            provenance: CheckpointProvenance {
                identifier: "model".into(),
                revision: "rev".into(),
                digest: "digest".into(),
                license: LicenseMetadata::Unknown,
            },
        }
    }
    fn config() -> DecoderConfig {
        DecoderConfig {
            vocab_size: 32,
            d_model: 16,
            num_heads: 4,
            num_layers: 3,
            d_ff: 64,
            max_seq_len: 128,
            norm_eps: 1e-5,
        }
    }
    fn caps(repack: bool) -> LoaderCapabilities {
        LoaderCapabilities {
            target_qkv_layout: QkvLayout::Separate,
            allow_qkv_repacking: repack,
        }
    }

    #[test]
    fn dimensions_do_not_authorize_loading() {
        let mut m = meta();
        m.architecture.activation = Activation::Relu;
        assert!(validate_decoder_dimensions(&m, &config()).is_ok());
        assert!(matches!(
            validate_loadable(&m, &arch(), &tok(), caps(false)),
            Err(CompatibilityError::NotLoadable(_))
        ));
    }
    #[test]
    fn invalid_local_metadata_and_dimensions_are_distinct() {
        let mut c = config();
        c.num_heads = 0;
        assert!(matches!(
            validate_decoder_dimensions(&meta(), &c),
            Err(CompatibilityError::InvalidLocalConfig(_))
        ));
        let mut m = meta();
        m.provenance.revision.clear();
        assert!(matches!(
            validate_decoder_dimensions(&m, &config()),
            Err(CompatibilityError::InvalidMetadata(_))
        ));
        let mut m = meta();
        m.architecture.d_ff += 1;
        assert!(matches!(
            validate_decoder_dimensions(&m, &config()),
            Err(CompatibilityError::DimensionMismatch(_))
        ));
    }
    #[test]
    fn rejects_invalid_rope_and_tokenizer_identity_in_stable_order() {
        let mut m = meta();
        m.architecture.position = PositionHandling::Rope(RopeSpec {
            rotary_dim: 3,
            interleaved: false,
            theta_bits: f64::NAN.to_bits(),
            scaling: None,
            max_positions: 1,
        });
        m.tokenizer.revision.clear();
        let CompatibilityError::InvalidMetadata(v) =
            validate_loadable(&m, &arch(), &tok(), caps(false)).unwrap_err()
        else {
            panic!()
        };
        assert_eq!(
            v.iter().map(|x| x.field).collect::<Vec<_>>(),
            [
                "position.rope.rotary_dim",
                "position.rope.theta",
                "position.rope.max_positions",
                "tokenizer.revision"
            ]
        );
    }
    #[test]
    fn qkv_conversion_is_explicit() {
        let mut m = meta();
        m.qkv_layout = QkvLayout::Fused;
        assert!(validate_loadable(&m, &arch(), &tok(), caps(true)).is_ok());
        assert!(matches!(
            validate_loadable(&m, &arch(), &tok(), caps(false)),
            Err(CompatibilityError::NotLoadable(_))
        ));
    }
    #[test]
    fn rejects_blank_special_token_name() {
        let mut m = meta();
        m.tokenizer.special_token_ids.push((" \t".into(), 2));
        let CompatibilityError::InvalidMetadata(v) =
            validate_loadable(&m, &arch(), &tok(), caps(false)).unwrap_err()
        else {
            panic!()
        };
        assert!(
            v.iter()
                .any(|m| m.field == "tokenizer.special_token_ids.name")
        );
    }
    #[test]
    fn rejects_out_of_range_special_token_id() {
        let mut tokenizer = tok();
        tokenizer
            .special_token_ids
            .push(("invalid".into(), tokenizer.vocab_size as u32));
        let CompatibilityError::InvalidMetadata(v) =
            validate_loadable(&meta(), &arch(), &tokenizer, caps(false)).unwrap_err()
        else {
            panic!()
        };
        assert!(
            v.iter()
                .any(|m| m.field == "tokenizer.special_token_ids.id")
        );
    }
    #[test]
    fn rejects_duplicate_special_token_name_with_conflicting_id() {
        let mut m = meta();
        m.tokenizer.special_token_ids.push(("eos".into(), 2));
        let CompatibilityError::InvalidMetadata(v) =
            validate_loadable(&m, &arch(), &tok(), caps(false)).unwrap_err()
        else {
            panic!()
        };
        assert!(
            v.iter()
                .any(|m| m.field == "tokenizer.special_token_ids.name")
        );
    }
    #[test]
    fn rejects_duplicate_special_token_id_with_different_names() {
        let mut malformed = tok();
        malformed.special_token_ids.push(("alias".into(), 1));
        let mut m = meta();
        m.tokenizer = malformed.clone();
        let CompatibilityError::InvalidMetadata(v) =
            validate_loadable(&m, &arch(), &malformed, caps(false)).unwrap_err()
        else {
            panic!()
        };
        assert!(
            v.iter()
                .any(|m| m.field == "tokenizer.special_token_ids.id")
        );
    }
    #[test]
    fn ownership_cannot_be_caller_forged() {
        assert_eq!(
            ParameterDestination::FreeTokenOutput(FreeTokenOutput::TiedTokenEmbeddingAlias)
                .ownership(),
            TensorOwnership::Checkpoint
        );
        for d in [
            ParameterDestination::ActionEmbedding,
            ParameterDestination::NeedEmbedding,
            ParameterDestination::StateEmbedding,
            ParameterDestination::ModalityEmbedding,
            ParameterDestination::StructuralHead,
            ParameterDestination::PointerHead,
        ] {
            assert_eq!(d.ownership(), TensorOwnership::NewlyInitialized);
        }
    }
    #[test]
    fn display_is_complete_and_ordered() {
        let e = CompatibilityError::dimensions(vec![mm("first", 1, 2), mm("second", "a", "b")]);
        assert_eq!(
            e.to_string(),
            "decoder dimensions differ; first: expected 1, found 2; second: expected \"a\", found \"b\""
        );
    }
}
