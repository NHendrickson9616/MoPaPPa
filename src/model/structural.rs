//! End-to-end inference for structural output heads.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::{
    bridge::{CandidateSet, ModelRoute, ModelStep},
    decoder::{Decoder, DecoderConfig, RoutedHeads, apply_legal_mask},
    embeddings::{EmbeddingConfig, InputEmbeddings},
    sequence::{CausalSequence, Position},
};

/// Shared embeddings and decoder followed by the fixed structural classifiers.
pub struct StructuralModel {
    embeddings: InputEmbeddings,
    decoder: Decoder,
    heads: RoutedHeads,
}

impl StructuralModel {
    /// Builds all components after checking their shared dimensions.
    pub fn new(
        embedding_config: EmbeddingConfig,
        decoder_config: DecoderConfig,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        embedding_config.validate()?;
        decoder_config.validate()?;
        if embedding_config.vocab_size != decoder_config.vocab_size {
            candle_core::bail!(
                "embedding vocab_size {} does not match decoder vocab_size {}",
                embedding_config.vocab_size,
                decoder_config.vocab_size
            )
        }
        if embedding_config.d_model != decoder_config.d_model {
            candle_core::bail!(
                "embedding d_model {} does not match decoder d_model {}",
                embedding_config.d_model,
                decoder_config.d_model
            )
        }
        if embedding_config.max_seq_len != decoder_config.max_seq_len {
            candle_core::bail!(
                "embedding max_seq_len {} does not match decoder max_seq_len {}",
                embedding_config.max_seq_len,
                decoder_config.max_seq_len
            )
        }

        let d_model = embedding_config.d_model;
        Ok(Self {
            embeddings: InputEmbeddings::new(embedding_config, vb.clone())?,
            decoder: Decoder::new(decoder_config, vb.clone())?,
            heads: RoutedHeads::new(d_model, vb)?,
        })
    }

    /// Scores the fixed head requested by `step`, masking every illegal class.
    ///
    /// The returned tensor retains the head's complete stable class vocabulary
    /// as `[1, candidate_count]`; this never compacts or translates candidate
    /// IDs.
    /// Low-level forward for a runner that owns both sequence and controller state.
    ///
    /// The sequence and step must come from the same current generation state.
    pub(crate) fn fixed_logits(
        &self,
        sequence: &CausalSequence,
        step: &ModelStep,
    ) -> Result<Tensor> {
        let last = sequence
            .positions()
            .last()
            .ok_or_else(|| candle_core::Error::Msg("sequence must not be empty".into()))?;
        let Position::Structural(position) = last.input() else {
            candle_core::bail!("last sequence position must be structural")
        };

        let (head, candidates) = match step {
            ModelStep::Complete => {
                candle_core::bail!("complete model step has no logits")
            }
            ModelStep::Needs {
                route: ModelRoute::DynamicPointer,
                ..
            } => candle_core::bail!("dynamic pointer route is not supported by fixed_logits"),
            ModelStep::Needs {
                route: ModelRoute::Value(_),
                ..
            } => candle_core::bail!("value route is not supported by fixed_logits"),
            ModelStep::Needs {
                route: ModelRoute::Fixed(_),
                candidates: CandidateSet::Symbols(_),
                ..
            } => candle_core::bail!("fixed route requires a fixed candidate set"),
            ModelStep::Needs {
                route: ModelRoute::Fixed(_),
                candidates: CandidateSet::Value(_),
                ..
            } => candle_core::bail!("fixed route requires a fixed candidate set"),
            ModelStep::Needs {
                route: ModelRoute::Fixed(head),
                candidates: CandidateSet::Fixed(candidates),
                ..
            } => (*head, candidates),
        };
        if position.output_head() != head {
            candle_core::bail!(
                "last structural output head {:?} does not match model step head {:?}",
                position.output_head(),
                head
            )
        }

        let candidate_count = head.fixed_candidate_count().ok_or_else(|| {
            candle_core::Error::Msg(
                "dynamic output head cannot be dispatched by fixed_logits".into(),
            )
        })?;
        if candidates.is_empty() {
            candle_core::bail!("fixed candidate set must not be empty")
        }
        let mut legal = vec![0u8; candidate_count];
        for candidate in candidates {
            let id = usize::from(candidate.id);
            if id >= candidate_count {
                candle_core::bail!(
                    "fixed candidate ID {} is out of range for {:?} (candidate count {})",
                    candidate.id,
                    head,
                    candidate_count
                )
            }
            if legal[id] != 0 {
                candle_core::bail!(
                    "duplicate fixed candidate ID {} for {:?}",
                    candidate.id,
                    head
                )
            }
            legal[id] = 1;
        }

        let embeddings = self.embeddings.forward(sequence)?;
        let hidden = self.decoder.last_hidden(&embeddings)?;
        let logits = self.heads.forward(head, &hidden)?;
        let legal = Tensor::from_vec(legal, (1, candidate_count), logits.device())?;
        apply_legal_mask(&logits, &legal)
    }

    /// Computes the decoder state for the last structural position.
    ///
    /// Runners retain this exact `[1, D]` state for declaration memory and use
    /// it to score dynamic pointers without a second decoder pass.
    pub(crate) fn last_hidden(&self, sequence: &CausalSequence) -> Result<Tensor> {
        let embeddings = self.embeddings.forward(sequence)?;
        self.decoder.last_hidden(&embeddings)
    }

    /// Dispatches and masks a fixed head from an already-computed last state.
    pub(crate) fn fixed_logits_from_hidden(
        &self,
        hidden: &Tensor,
        step: &ModelStep,
    ) -> Result<Tensor> {
        fixed_logits_from_hidden(&self.heads, hidden, step)
    }
}

pub(crate) fn fixed_logits_from_hidden(
    heads: &RoutedHeads,
    hidden: &Tensor,
    step: &ModelStep,
) -> Result<Tensor> {
    let (head, candidates) = fixed_request(step)?;
    let candidate_count = head
        .fixed_candidate_count()
        .expect("fixed output head has a fixed vocabulary");
    let mut legal = vec![0u8; candidate_count];
    for candidate in candidates {
        let id = usize::from(candidate.id);
        if id >= candidate_count {
            candle_core::bail!("fixed candidate ID {} is out of range", candidate.id)
        }
        if legal[id] != 0 {
            candle_core::bail!("duplicate fixed candidate ID {}", candidate.id)
        }
        legal[id] = 1;
    }
    let logits = heads.forward(head, hidden)?;
    let legal = Tensor::from_vec(legal, (1, candidate_count), logits.device())?;
    apply_legal_mask(&logits, &legal)
}

fn fixed_request(
    step: &ModelStep,
) -> Result<(
    crate::training::OutputHead,
    &[super::bridge::FixedCandidate],
)> {
    match step {
        ModelStep::Needs {
            route: ModelRoute::Fixed(head),
            candidates: CandidateSet::Fixed(candidates),
            ..
        } if !candidates.is_empty() => Ok((*head, candidates)),
        _ => candle_core::bail!("model step is not a nonempty fixed request"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    use crate::{
        controller::{Action, BlockAction, Need, RootAction, TraceStep},
        decode::{Decision, DecodeController, RootChoice},
        model::{
            bridge::{ValueRequest, step},
            sequence::TokenizerIdentity,
        },
        training::{OutputHead, sequence_from_trace},
    };

    fn tokenizer() -> TokenizerIdentity {
        TokenizerIdentity::new("test", "revision", "vocabulary").unwrap()
    }

    fn embedding_config() -> EmbeddingConfig {
        EmbeddingConfig {
            vocab_size: 16,
            d_model: 8,
            max_seq_len: 8,
            tokenizer: tokenizer(),
        }
    }

    fn decoder_config() -> DecoderConfig {
        DecoderConfig {
            vocab_size: 16,
            d_model: 8,
            num_heads: 2,
            num_layers: 1,
            d_ff: 16,
            max_seq_len: 8,
            norm_eps: 1e-5,
        }
    }

    fn model<'a>(vars: &'a VarMap, device: &'a Device) -> StructuralModel {
        StructuralModel::new(
            embedding_config(),
            decoder_config(),
            VarBuilder::from_varmap(vars, DType::F32, device),
        )
        .unwrap()
    }

    fn sequence(steps: Vec<TraceStep>) -> CausalSequence {
        let training = sequence_from_trace(steps).unwrap();
        CausalSequence::from_training(Vec::new(), tokenizer(), &training).unwrap()
    }

    fn trace(need: Need, action: Action) -> TraceStep {
        TraceStep { need, action }
    }

    #[test]
    fn live_root_step_runs_the_complete_fixed_path() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let model = model(&vars, &device);
        let controller = DecodeController::new();
        let step = step(&controller);
        let sequence = sequence(vec![trace(Need::Root, Action::Root(RootAction::Module))]);

        let logits = model.fixed_logits(&sequence, &step)?.to_vec2::<f32>()?;
        assert_eq!((logits.len(), logits[0].len()), (1, 2));
        assert!(logits[0].iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn block_fragment_masks_only_the_illegal_empty_statement_class() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let model = model(&vars, &device);
        let mut controller = DecodeController::new();
        controller
            .apply(Decision::Root(RootChoice::BlockFragment))
            .unwrap();
        let step = step(&controller);
        let sequence = sequence(vec![
            trace(Need::Root, Action::Root(RootAction::BlockFragment)),
            trace(
                Need::BlockEntryOrEnd,
                Action::Block(BlockAction::EndAndYield),
            ),
        ]);

        let logits = model.fixed_logits(&sequence, &step)?.to_vec2::<f32>()?;
        assert_eq!((logits.len(), logits[0].len()), (1, 4));
        assert_eq!(logits[0][0], f32::NEG_INFINITY);
        assert!(logits[0][1..].iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn rejects_mismatched_and_non_fixed_routes() {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let model = model(&vars, &device);
        let sequence = sequence(vec![trace(Need::Root, Action::Root(RootAction::Module))]);
        let mut controller = DecodeController::new();
        controller
            .apply(Decision::Root(RootChoice::BlockFragment))
            .unwrap();
        assert!(model.fixed_logits(&sequence, &step(&controller)).is_err());

        let state = step(&DecodeController::new()).state().unwrap();
        let route = |route, candidates| ModelStep::Needs {
            need: Need::Root,
            route,
            state,
            candidates,
        };
        assert!(
            model
                .fixed_logits(
                    &sequence,
                    &route(
                        ModelRoute::DynamicPointer,
                        CandidateSet::Symbols(Vec::new())
                    )
                )
                .is_err()
        );
        assert!(
            model
                .fixed_logits(
                    &sequence,
                    &route(
                        ModelRoute::Value(ValueRequest::Integer),
                        CandidateSet::Value(ValueRequest::Integer)
                    )
                )
                .is_err()
        );
        assert!(model.fixed_logits(&sequence, &ModelStep::Complete).is_err());
    }

    #[test]
    fn constructor_rejects_each_shared_dimension_mismatch() {
        let device = Device::Cpu;
        for changed in 0..3 {
            let vars = VarMap::new();
            let embeddings = embedding_config();
            let mut decoder = decoder_config();
            match changed {
                0 => decoder.vocab_size += 1,
                1 => decoder.d_model += 2,
                _ => decoder.max_seq_len += 1,
            }
            assert!(
                StructuralModel::new(
                    embeddings,
                    decoder,
                    VarBuilder::from_varmap(&vars, DType::F32, &device)
                )
                .is_err()
            );
        }
    }

    #[test]
    fn complete_model_keeps_existing_parameter_groups() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let _model = model(&vars, &device);
        let path = std::env::temp_dir().join(format!(
            "mopappa-structural-model-groups-{}-{}.safetensors",
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
            .filter(|name| {
                ["pretrained.", "backbone.", "structural."]
                    .iter()
                    .any(|prefix| name.starts_with(prefix))
            })
            .collect();
        assert_eq!(names.len(), vars.all_vars().len());
        for prefix in ["pretrained.", "backbone.", "structural."] {
            assert!(
                names.iter().any(|name| name.starts_with(prefix)),
                "{prefix}"
            );
        }
        Ok(())
    }

    #[test]
    fn block_controller_fixture_has_the_expected_stable_ids() {
        let mut controller = DecodeController::new();
        controller
            .apply(Decision::Root(RootChoice::BlockFragment))
            .unwrap();
        let block = step(&controller);
        let CandidateSet::Fixed(candidates) = block.candidates().unwrap() else {
            panic!("block route must be fixed")
        };
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.id)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(block.route(), Some(ModelRoute::Fixed(OutputHead::Block)));
    }
}
