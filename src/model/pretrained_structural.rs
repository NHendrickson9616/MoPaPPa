//! Frozen pretrained LLaMA backbone with trainable structural inputs and heads.

use candle_core::{DType, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;

use super::runner::GenerationModel;
use super::{
    bridge::{ModelRoute, ModelStep},
    decoder::RoutedHeads,
    embeddings::StructuralEmbeddings,
    pretrained_llama::{Cache, Config, Llama},
    sequence::{
        CausalSequence, Position, SEQUENCE_PROTOCOL_VERSION, SupportedMetadata, TokenizerIdentity,
    },
    structural::fixed_logits_from_hidden,
};
use crate::training::TRAINING_SCHEMA_VERSION;

/// A pretrained backbone whose only trainable tensors live in a separate
/// `structural.*` [`candle_nn::VarMap`].
pub struct PretrainedStructuralModel {
    backbone: Llama,
    config: Config,
    dtype: DType,
    cache: Cache,
    embeddings: StructuralEmbeddings,
    heads: RoutedHeads,
    tokenizer: TokenizerIdentity,
}

impl PretrainedStructuralModel {
    pub fn new(
        config: Config,
        tokenizer: TokenizerIdentity,
        dtype: DType,
        backbone_vb: VarBuilder<'_>,
        structural_vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let backbone = Llama::load(backbone_vb, &config)?;
        let cache = Cache::new(false, dtype, &config, structural_vb.device())?;
        let embeddings = StructuralEmbeddings::new(
            config.hidden_size,
            structural_vb.pp("structural").pp("embeddings"),
            dtype,
        )?;
        let heads = RoutedHeads::new(config.hidden_size, structural_vb)?;
        Ok(Self {
            backbone,
            config,
            dtype,
            cache,
            embeddings,
            heads,
            tokenizer,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// Scores the requested fixed head. Pointer and value routes are unsupported.
    pub fn fixed_logits(&mut self, sequence: &CausalSequence, step: &ModelStep) -> Result<Tensor> {
        let last = sequence
            .positions()
            .last()
            .ok_or_else(|| candle_core::Error::Msg("sequence must not be empty".into()))?;
        let Position::Structural(position) = last.input() else {
            candle_core::bail!("last sequence position must be structural")
        };
        let ModelStep::Needs {
            route: ModelRoute::Fixed(head),
            ..
        } = step
        else {
            candle_core::bail!("pretrained structural model supports fixed heads only")
        };
        if position.output_head() != *head {
            candle_core::bail!("last structural output head does not match model step head")
        }
        let hidden = self.last_hidden(sequence)?;
        self.fixed_logits_from_hidden(&hidden, step)
    }

    fn last_hidden(&mut self, sequence: &CausalSequence) -> Result<Tensor> {
        sequence
            .validate_supported(&SupportedMetadata {
                protocol_version: SEQUENCE_PROTOCOL_VERSION,
                training_schema_version: TRAINING_SCHEMA_VERSION,
                tokenizer: self.tokenizer.clone(),
            })
            .map_err(|error| {
                candle_core::Error::Msg(format!("unsupported causal sequence: {error:?}"))
            })?;
        let positions = sequence.positions();
        let last = positions
            .last()
            .ok_or_else(|| candle_core::Error::Msg("sequence must not be empty".into()))?;
        let Position::Structural(_) = last.input() else {
            candle_core::bail!("last sequence position must be structural")
        };
        if positions.len() > self.config.max_position_embeddings {
            candle_core::bail!("sequence exceeds pretrained maximum position count")
        }

        let device = self.cache_device().clone();
        let mut rows = Vec::with_capacity(positions.len());
        for item in positions {
            let row = match item.input() {
                Position::EnglishToken(id) => {
                    if *id as usize >= self.config.vocab_size {
                        candle_core::bail!("token ID {id} is outside pretrained vocabulary")
                    }
                    self.backbone
                        .embed(&Tensor::new(&[[*id]], &device)?)?
                        .squeeze(0)?
                }
                Position::Structural(position) => self
                    .embeddings
                    .forward(position, self.config.max_position_embeddings)?,
                Position::SymbolNameToken(_) | Position::FreeToken(_) => {
                    candle_core::bail!("reserved unsupported position")
                }
            };
            rows.push(row);
        }
        let input = Tensor::cat(&rows, 0)?.unsqueeze(0)?;
        self.cache = Cache::new(false, self.dtype, &self.config, &device)?;
        let hidden = self.backbone.forward_hidden(&input, 0, &mut self.cache)?;
        hidden.i((.., positions.len() - 1, ..))?.contiguous()
    }

    fn fixed_logits_from_hidden(&self, hidden: &Tensor, step: &ModelStep) -> Result<Tensor> {
        fixed_logits_from_hidden(&self.heads, hidden, step)
    }

    fn cache_device(&self) -> &candle_core::Device {
        // Every structural tensor is created on the same device as the cache.
        // The head output is a convenient stable device owner.
        self.embeddings.device()
    }
}

impl GenerationModel for PretrainedStructuralModel {
    fn last_hidden(&mut self, sequence: &CausalSequence) -> Result<Tensor> {
        PretrainedStructuralModel::last_hidden(self, sequence)
    }

    fn fixed_logits_from_hidden(&mut self, hidden: &Tensor, step: &ModelStep) -> Result<Tensor> {
        PretrainedStructuralModel::fixed_logits_from_hidden(self, hidden, step)
    }
}
