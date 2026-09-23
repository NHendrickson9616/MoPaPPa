//! Composition of one validated causal sequence into decoder-width inputs.
//!
//! `OutputHead` is already the canonical one-to-one representation of the
//! current `Need`, so structural rows intentionally have no second Need table.
//! Declaration and pointer identities are likewise deferred to future
//! declaration memory rather than treated as dense categorical IDs here.
//! The learned external-position table belongs to the checkpoint/pretrained
//! family. Payload projection is new structural state/payload machinery.

use candle_core::{DType, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder, embedding, linear};

use super::{
    embedding_ids::{
        ENGLISH_MODALITY_ID, MODALITY_COUNT, OUTPUT_HEAD_COUNT, PREVIOUS_ACTION_COUNT,
        STRUCTURAL_MODALITY_ID, literal_payload_features, output_head_id, previous_action_id,
    },
    sequence::{
        CausalSequence, Position, PreviousAction, SEQUENCE_PROTOCOL_VERSION, SupportedMetadata,
        TokenizerIdentity,
    },
};
use crate::controller::Action;
use crate::training::TRAINING_SCHEMA_VERSION;

const STATE_FEATURE_COUNT: usize = 2;
const PAYLOAD_FEATURE_COUNT: usize = 13;

/// Dimensions and tokenizer identity accepted by an [`InputEmbeddings`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddingConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub max_seq_len: usize,
    pub tokenizer: TokenizerIdentity,
}

impl EmbeddingConfig {
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("vocab_size", self.vocab_size),
            ("d_model", self.d_model),
            ("max_seq_len", self.max_seq_len),
        ] {
            if value == 0 {
                candle_core::bail!("{name} must be greater than zero")
            }
        }
        Ok(())
    }
}

/// Learned composer producing a full-width `[1, T, D]` tensor directly.
pub struct InputEmbeddings {
    config: EmbeddingConfig,
    token: Embedding,
    position: Embedding,
    structural: StructuralEmbeddings,
}

/// Learned features shared by every structural position composer.
///
/// Token and absolute-position embeddings deliberately do not belong here.
pub(crate) struct StructuralEmbeddings {
    modality: Embedding,
    output_head: Embedding,
    previous_action: Embedding,
    state: Linear,
    payload: Linear,
    dtype: DType,
    d_model: usize,
}

impl InputEmbeddings {
    pub fn new(config: EmbeddingConfig, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let pretrained = vb.pp("pretrained");
        let structural = vb.pp("structural").pp("embeddings");
        Ok(Self {
            token: embedding(config.vocab_size, config.d_model, pretrained.pp("token"))?,
            position: embedding(
                config.max_seq_len,
                config.d_model,
                pretrained.pp("position"),
            )?,
            structural: StructuralEmbeddings::new(config.d_model, structural, vb.dtype())?,
            config,
        })
    }

    /// Composes one sequence. Batching, padding, and symbol memory are outside
    /// this boundary.
    pub fn forward(&self, sequence: &CausalSequence) -> Result<Tensor> {
        sequence
            .validate_supported(&SupportedMetadata {
                protocol_version: SEQUENCE_PROTOCOL_VERSION,
                training_schema_version: TRAINING_SCHEMA_VERSION,
                tokenizer: self.config.tokenizer.clone(),
            })
            .map_err(|error| {
                candle_core::Error::Msg(format!("unsupported causal sequence: {error:?}"))
            })?;
        let length = sequence.positions().len();
        if length == 0 {
            candle_core::bail!("sequence must contain at least one position")
        }
        if length > self.config.max_seq_len {
            candle_core::bail!(
                "sequence length {length} exceeds max_seq_len {}",
                self.config.max_seq_len
            )
        }

        let device = self.position.embeddings().device();
        let english_modality = self.structural.modality(ENGLISH_MODALITY_ID)?;
        let mut rows = Vec::with_capacity(length);
        for (index, item) in sequence.positions().iter().enumerate() {
            let position = self
                .position
                .forward(&Tensor::new(&[index as u32], device)?)?;
            let row = match item.input() {
                Position::EnglishToken(token) => {
                    if *token as usize >= self.config.vocab_size {
                        candle_core::bail!(
                            "token ID {token} at position {index} is outside vocabulary {}",
                            self.config.vocab_size
                        )
                    }
                    (&self.token.forward(&Tensor::new(&[*token], device)?)? + &position)?
                        .broadcast_add(&english_modality)?
                }
                Position::Structural(input) => {
                    (&position + self.structural.forward(input, self.config.max_seq_len)?)?
                }
                Position::SymbolNameToken(_) | Position::FreeToken(_) => {
                    candle_core::bail!("reserved unsupported position at sequence position {index}")
                }
            };
            rows.push(row);
        }
        Tensor::cat(&rows, 0)?.unsqueeze(0)
    }
}

impl StructuralEmbeddings {
    pub(crate) fn new(d_model: usize, vb: VarBuilder<'_>, dtype: DType) -> Result<Self> {
        Ok(Self {
            modality: embedding(MODALITY_COUNT as usize, d_model, vb.pp("modality"))?,
            output_head: embedding(OUTPUT_HEAD_COUNT as usize, d_model, vb.pp("output_head"))?,
            previous_action: embedding(
                PREVIOUS_ACTION_COUNT as usize,
                d_model,
                vb.pp("previous_action"),
            )?,
            state: linear(STATE_FEATURE_COUNT, d_model, vb.pp("state"))?,
            payload: linear(PAYLOAD_FEATURE_COUNT, d_model, vb.pp("payload"))?,
            dtype,
            d_model,
        })
    }

    fn modality(&self, id: u32) -> Result<Tensor> {
        self.modality
            .forward(&Tensor::new(&[id], self.modality.embeddings().device())?)
    }

    pub(crate) fn device(&self) -> &candle_core::Device {
        self.modality.embeddings().device()
    }

    /// Composes one structural position as `[1, D]`.
    pub(crate) fn forward(
        &self,
        input: &super::sequence::StructuralPosition,
        max_seq_len: usize,
    ) -> Result<Tensor> {
        let device = self.modality.embeddings().device();
        let modality = self.modality(STRUCTURAL_MODALITY_ID)?;
        let head = self.output_head.forward(&Tensor::new(
            &[output_head_id(input.output_head())],
            device,
        )?)?;
        let previous = self.previous_action.forward(&Tensor::new(
            &[previous_action_id(input.previous_action())],
            device,
        )?)?;
        let state_features = Tensor::new(&normalized_state(input.state(), max_seq_len), device)?
            .to_dtype(self.dtype)?
            .reshape((1, STATE_FEATURE_COUNT))?;
        let state = self.state.forward(&state_features)?;
        let payload = self.payload.forward(
            &Tensor::new(&payload_features(input.previous_action()), device)?
                .to_dtype(self.dtype)?
                .reshape((1, PAYLOAD_FEATURE_COUNT))?,
        )?;
        let row = ((((&modality + &head)? + &previous)? + state)? + payload)?;
        debug_assert_eq!(row.dims(), &[1, self.d_model]);
        Ok(row)
    }
}

fn normalized_state(
    state: super::sequence::ControllerStateFeatures,
    max_seq_len: usize,
) -> [f32; STATE_FEATURE_COUNT] {
    let scale = (max_seq_len as f32).ln_1p();
    [
        ((state.completed_actions as f32).ln_1p() / scale).min(1.0),
        ((state.dynamic_candidate_count as f32).ln_1p() / scale).min(1.0),
    ]
}

fn payload_features(previous: &PreviousAction) -> [f32; PAYLOAD_FEATURE_COUNT] {
    let literal = match previous {
        PreviousAction::Action(Action::Literal(literal)) => Some(literal),
        _ => None,
    };
    let Some(literal) = literal else {
        return [0.0; PAYLOAD_FEATURE_COUNT];
    };
    let value = literal_payload_features(literal);
    let mut features = [0.0; PAYLOAD_FEATURE_COUNT];
    features[0] = value.is_integer;
    features[1..9].copy_from_slice(&value.integer);
    features[9] = value.is_bool;
    features[10] = value.bool_value;
    features[11] = value.is_string;
    features[12] = value.string_byte_length;
    features
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    use crate::{
        controller::{Action, Need, RootAction},
        model::decoder::{Decoder, DecoderConfig},
        training::{
            Annotation, FixedTarget, Observation, OutputHead, SchemaVersion,
            TRAINING_SCHEMA_VERSION, Target, TeacherForcingRecord, TrainingSequence,
        },
    };

    fn tokenizer(name: &str) -> TokenizerIdentity {
        TokenizerIdentity::new(name, "revision", "vocabulary").unwrap()
    }

    fn config(tokenizer: TokenizerIdentity) -> EmbeddingConfig {
        EmbeddingConfig {
            vocab_size: 16,
            d_model: 8,
            max_seq_len: 8,
            tokenizer,
        }
    }

    fn training(actions: Vec<RootAction>) -> TrainingSequence {
        TrainingSequence {
            schema_version: TRAINING_SCHEMA_VERSION,
            records: actions
                .iter()
                .enumerate()
                .map(|(index, action)| TeacherForcingRecord {
                    step_index: index,
                    need: Need::Root,
                    history_end: index,
                    output_head: OutputHead::Root,
                    target: Target::Fixed(FixedTarget::Root(*action)),
                    observation: Observation::None,
                    annotation: Annotation::None,
                })
                .collect(),
            actions: actions.into_iter().map(Action::Root).collect(),
        }
    }

    fn sequence(
        prefix: Vec<u32>,
        tokenizer: TokenizerIdentity,
        actions: Vec<RootAction>,
    ) -> CausalSequence {
        CausalSequence::from_training(prefix, tokenizer, &training(actions)).unwrap()
    }

    fn composer<'a>(
        vars: &'a VarMap,
        device: &'a Device,
        tokenizer: TokenizerIdentity,
    ) -> InputEmbeddings {
        InputEmbeddings::new(
            config(tokenizer),
            VarBuilder::from_varmap(vars, DType::F32, device),
        )
        .unwrap()
    }

    #[test]
    fn config_rejects_every_zero_dimension() {
        let base = config(tokenizer("test"));
        for invalid in [
            EmbeddingConfig {
                vocab_size: 0,
                ..base.clone()
            },
            EmbeddingConfig {
                d_model: 0,
                ..base.clone()
            },
            EmbeddingConfig {
                max_seq_len: 0,
                ..base.clone()
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }

    #[test]
    fn tokenizer_token_and_position_bounds_are_enforced() {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let expected = tokenizer("expected");
        let embeddings = composer(&vars, &device, expected.clone());

        assert!(
            embeddings
                .forward(&sequence(vec![], expected.clone(), vec![]))
                .is_err()
        );
        assert!(
            embeddings
                .forward(&sequence(vec![1], tokenizer("other"), vec![]))
                .is_err()
        );
        let mut unsupported_schema = training(vec![]);
        unsupported_schema.schema_version = SchemaVersion(TRAINING_SCHEMA_VERSION.0 + 1);
        let unsupported_schema =
            CausalSequence::from_training(vec![1], expected.clone(), &unsupported_schema).unwrap();
        assert!(embeddings.forward(&unsupported_schema).is_err());
        assert!(
            embeddings
                .forward(&sequence(vec![16], expected.clone(), vec![]))
                .is_err()
        );
        assert!(
            embeddings
                .forward(&sequence(vec![1; 9], expected, vec![]))
                .is_err()
        );
    }

    #[test]
    fn english_and_structural_rows_share_full_decoder_width() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let identity = tokenizer("test");
        let output = composer(&vars, &device, identity.clone()).forward(&sequence(
            vec![1, 2],
            identity,
            vec![RootAction::Module],
        ))?;
        assert_eq!(output.dims3()?, (1, 3, 8));
        Ok(())
    }

    #[test]
    fn changing_an_english_token_changes_only_its_row() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let identity = tokenizer("test");
        let embeddings = composer(&vars, &device, identity.clone());
        let left = embeddings.forward(&sequence(
            vec![1, 2],
            identity.clone(),
            vec![RootAction::Module],
        ))?;
        let right =
            embeddings.forward(&sequence(vec![1, 3], identity, vec![RootAction::Module]))?;
        assert_eq!(
            left.narrow(1, 0, 1)?.to_vec3::<f32>()?,
            right.narrow(1, 0, 1)?.to_vec3::<f32>()?
        );
        assert_ne!(
            left.narrow(1, 1, 1)?.to_vec3::<f32>()?,
            right.narrow(1, 1, 1)?.to_vec3::<f32>()?
        );
        assert_eq!(
            left.narrow(1, 2, 1)?.to_vec3::<f32>()?,
            right.narrow(1, 2, 1)?.to_vec3::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn changing_previous_fixed_action_changes_next_structural_row() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let identity = tokenizer("test");
        let embeddings = composer(&vars, &device, identity.clone());
        let module = embeddings.forward(&sequence(
            vec![],
            identity.clone(),
            vec![RootAction::Module, RootAction::Module],
        ))?;
        let fragment = embeddings.forward(&sequence(
            vec![],
            identity,
            vec![RootAction::BlockFragment, RootAction::Module],
        ))?;
        assert_eq!(
            module.narrow(1, 0, 1)?.to_vec3::<f32>()?,
            fragment.narrow(1, 0, 1)?.to_vec3::<f32>()?
        );
        assert_ne!(
            module.narrow(1, 1, 1)?.to_vec3::<f32>()?,
            fragment.narrow(1, 1, 1)?.to_vec3::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn parameter_ownership_prefixes_are_exact() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let _ = composer(&vars, &device, tokenizer("test"));
        let path = std::env::temp_dir().join(format!(
            "mopappa-input-embeddings-{}-{}.safetensors",
            std::process::id(),
            vars.all_vars().len()
        ));
        vars.save(&path)?;
        let bytes = std::fs::read(&path)?;
        std::fs::remove_file(path)?;
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header = std::str::from_utf8(&bytes[8..8 + header_len]).unwrap();
        let names: Vec<_> = header
            .split('"')
            .filter(|name| name.starts_with("pretrained.") || name.starts_with("structural."))
            .collect();
        let mut names = names;
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "pretrained.position.weight",
                "pretrained.token.weight",
                "structural.embeddings.modality.weight",
                "structural.embeddings.output_head.weight",
                "structural.embeddings.payload.bias",
                "structural.embeddings.payload.weight",
                "structural.embeddings.previous_action.weight",
                "structural.embeddings.state.bias",
                "structural.embeddings.state.weight",
            ]
        );
        Ok(())
    }

    #[test]
    fn state_scaling_and_payload_zero_are_explicit() {
        use crate::model::sequence::ControllerStateFeatures;
        let zero = normalized_state(
            ControllerStateFeatures {
                completed_actions: 0,
                dynamic_candidate_count: 0,
            },
            128,
        );
        let one = normalized_state(
            ControllerStateFeatures {
                completed_actions: 1,
                dynamic_candidate_count: 1,
            },
            128,
        );
        let realistic = normalized_state(
            ControllerStateFeatures {
                completed_actions: 64,
                dynamic_candidate_count: 16,
            },
            128,
        );
        assert_eq!(zero, [0.0, 0.0]);
        assert!(one[0] > 0.0 && one[1] > 0.0);
        assert!(realistic[0] > one[0] && realistic[1] > one[1]);
        assert_eq!(
            payload_features(&PreviousAction::Bos),
            [0.0; PAYLOAD_FEATURE_COUNT]
        );
    }

    #[test]
    fn f64_builder_supports_structural_rows() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let identity = tokenizer("f64");
        let embeddings = InputEmbeddings::new(
            config(identity.clone()),
            VarBuilder::from_varmap(&vars, DType::F64, &device),
        )?;
        let output = embeddings.forward(&sequence(vec![1], identity, vec![RootAction::Module]))?;
        assert_eq!(output.dtype(), DType::F64);
        assert_eq!(output.dims3()?, (1, 2, 8));
        Ok(())
    }

    #[test]
    fn composed_input_runs_through_tiny_decoder() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let identity = tokenizer("test");
        let input = composer(&vars, &device, identity.clone()).forward(&sequence(
            vec![1],
            identity,
            vec![RootAction::Module],
        ))?;
        let decoder = Decoder::new(
            DecoderConfig {
                vocab_size: 16,
                d_model: 8,
                num_heads: 2,
                num_layers: 1,
                d_ff: 16,
                max_seq_len: 8,
                norm_eps: 1e-5,
            },
            VarBuilder::from_varmap(&vars, DType::F32, &device),
        )?;
        assert_eq!(decoder.forward(&input)?.dims3()?, (1, 2, 8));
        Ok(())
    }
}
