//! Small, executable training experiments.
//!
//! This module is intentionally explicit rather than a general training
//! framework. It demonstrates that the structural parameters can learn while
//! the randomly initialized token embeddings and decoder remain frozen.

use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::{AdamW, Optimizer, VarBuilder, VarMap, loss};

use crate::{
    decode::DecodeController,
    model::{
        bridge::{ModelStep, step},
        decoder::DecoderConfig,
        embeddings::EmbeddingConfig,
        sequence::{CausalSequence, PreviousAction, StructuralPosition, TokenizerIdentity},
        structural::StructuralModel,
    },
};

/// Outcome of the two-prompt frozen-backbone experiment.
#[derive(Clone, Debug)]
pub struct TinyOverfitReport {
    pub initial_loss: f32,
    pub final_loss: f32,
    pub iterations: usize,
    pub trainable_vars: usize,
    pub total_vars: usize,
    pub predictions: [u32; 2],
}

/// Overfits two root decisions while optimizing only `structural.*` variables.
pub fn run_tiny_overfit() -> Result<TinyOverfitReport> {
    run_tiny_overfit_on(&Device::Cpu)
}

/// Runs the tiny frozen-backbone experiment on the selected device.
pub fn run_tiny_overfit_on(device: &Device) -> Result<TinyOverfitReport> {
    let tokenizer = TokenizerIdentity::new("tiny-overfit", "1", "eight-tokens")
        .map_err(|error| candle_core::Error::Msg(format!("{error:?}")))?;
    let vars = VarMap::new();
    let model = StructuralModel::new(
        EmbeddingConfig {
            vocab_size: 8,
            d_model: 16,
            max_seq_len: 8,
            tokenizer: tokenizer.clone(),
        },
        DecoderConfig {
            vocab_size: 8,
            d_model: 16,
            num_heads: 4,
            num_layers: 1,
            d_ff: 32,
            max_seq_len: 8,
            norm_eps: 1e-5,
        },
        VarBuilder::from_varmap(&vars, DType::F32, device),
    )?;

    // Both examples ask the untouched controller for its initial Root step.
    // from_inference builds the same causal representation used by generation.
    let root_step = step(&DecodeController::new());
    let examples = [
        inference_sequence(&[1], &tokenizer, &root_step)?,
        inference_sequence(&[2], &tokenizer, &root_step)?,
    ];
    let targets = [Tensor::new(&[0u32], device)?, Tensor::new(&[1u32], device)?];

    let (trainable, total_vars) = structural_variables(&vars);
    let trainable_vars = trainable.len();
    let mut optimizer = AdamW::new_lr(trainable, 3e-2)?;

    let initial_loss = mean_loss(&model, &examples, &root_step, &targets)?.to_scalar::<f32>()?;
    ensure_finite(initial_loss, 0)?;
    println!("iteration   0 loss {initial_loss:.6}");

    let mut iterations = 0;
    for iteration in 1..=300 {
        let loss = mean_loss(&model, &examples, &root_step, &targets)?;
        let current_loss = loss.to_scalar::<f32>()?;
        ensure_finite(current_loss, iteration)?;
        optimizer.backward_step(&loss)?;
        iterations = iteration;

        if iteration % 25 == 0 || current_loss < 0.01 {
            println!("iteration {iteration:3} loss {current_loss:.6}");
        }
        if current_loss < 0.01 {
            break;
        }
    }

    // Recompute after the last optimizer step so the report describes the final
    // parameters rather than the parameters immediately before that step.
    let final_loss = mean_loss(&model, &examples, &root_step, &targets)?.to_scalar::<f32>()?;
    ensure_finite(final_loss, iterations)?;
    let predictions = [
        prediction(&model, &examples[0], &root_step)?,
        prediction(&model, &examples[1], &root_step)?,
    ];

    Ok(TinyOverfitReport {
        initial_loss,
        final_loss,
        iterations,
        trainable_vars,
        total_vars,
        predictions,
    })
}

fn inference_sequence(
    english: &[u32],
    tokenizer: &TokenizerIdentity,
    model_step: &ModelStep,
) -> Result<CausalSequence> {
    let position = StructuralPosition::from_model_step(model_step, PreviousAction::Bos, 0)
        .map_err(|error| candle_core::Error::Msg(format!("{error:?}")))?;
    Ok(CausalSequence::from_inference(
        english,
        tokenizer,
        &[],
        position,
    ))
}

fn structural_variables(vars: &VarMap) -> (Vec<Var>, usize) {
    let data = vars.data().lock().expect("VarMap mutex poisoned");
    let trainable = data
        .iter()
        .filter(|(name, _)| name.starts_with("structural."))
        .map(|(_, var)| var.clone())
        .collect();
    (trainable, data.len())
}

fn mean_loss(
    model: &StructuralModel,
    examples: &[CausalSequence; 2],
    step: &ModelStep,
    targets: &[Tensor; 2],
) -> Result<Tensor> {
    let first = loss::cross_entropy(&model.fixed_logits(&examples[0], step)?, &targets[0])?;
    let second = loss::cross_entropy(&model.fixed_logits(&examples[1], step)?, &targets[1])?;
    (first + second)? / 2f64
}

fn prediction(model: &StructuralModel, sequence: &CausalSequence, step: &ModelStep) -> Result<u32> {
    model
        .fixed_logits(sequence, step)?
        .argmax(1)?
        .to_vec1::<u32>()
        .map(|values| values[0])
}

fn ensure_finite(loss: f32, iteration: usize) -> Result<()> {
    if !loss.is_finite() {
        candle_core::bail!("loss became non-finite at iteration {iteration}: {loss}")
    }
    Ok(())
}
