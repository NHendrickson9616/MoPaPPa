use candle_core::{D, DType, Tensor};
use candle_nn::{AdamW, Optimizer, VarBuilder, VarMap, loss};
use candle_transformers::models::llama::LlamaConfig;
use hf_hub::{Repo, RepoType, api::sync::Api};
use mopappa::{
    decode::DecodeController,
    engine::device::{device_name, selected_device},
    model::{
        bridge::{ModelStep, step},
        pretrained_structural::PretrainedStructuralModel,
        sequence::{CausalSequence, PreviousAction, StructuralPosition, TokenizerIdentity},
    },
};
use tokenizers::Tokenizer;

const MODEL_ID: &str = "HuggingFaceTB/SmolLM2-135M";
const REVISION: &str = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2";
const PROMPTS: [&str; 2] = ["create a Rust module", "create a Rust block"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = selected_device()?;
    // F32 is intentional: Candle CPU matmul cannot train BF16, and using one
    // dtype on both CPU and CUDA keeps structural/backbone matmuls compatible.
    let dtype = DType::F32;
    let repo = Api::new()?.repo(Repo::with_revision(
        MODEL_ID.into(),
        RepoType::Model,
        REVISION.into(),
    ));
    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;
    let weights_path = repo.get("model.safetensors")?;
    let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
    let config = config.into_config(false);
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|error| format!("cannot load tokenizer: {error}"))?;
    let identity = TokenizerIdentity::new(MODEL_ID, REVISION, "tokenizer.json")
        .map_err(|error| format!("invalid tokenizer identity: {error:?}"))?;
    let root_step = step(&DecodeController::new());
    let examples = PROMPTS
        .map(|prompt| inference_sequence(&tokenizer, prompt, &identity, &root_step))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let targets = [
        Tensor::new(&[0u32], &device)?,
        Tensor::new(&[1u32], &device)?,
    ];

    // The mmap-backed builder is immutable and cannot place backbone tensors
    // in the independently owned trainable VarMap.
    let backbone_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)? };
    let structural_vars = VarMap::new();
    let structural_vb = VarBuilder::from_varmap(&structural_vars, dtype, &device);
    let mut model =
        PretrainedStructuralModel::new(config, identity, dtype, backbone_vb, structural_vb)?;
    let variables = structural_vars.all_vars();
    let trainable_count = variables.len();
    let mut optimizer = AdamW::new_lr(variables, 1e-2)?;

    let initial_loss =
        mean_loss(&mut model, &examples, &root_step, &targets)?.to_scalar::<f32>()?;
    println!("iteration   0 loss {initial_loss:.6}");
    let mut iterations = 0;
    for iteration in 1..=300 {
        let current = mean_loss(&mut model, &examples, &root_step, &targets)?;
        let value = current.to_scalar::<f32>()?;
        if !value.is_finite() {
            return Err(format!("loss became non-finite at iteration {iteration}").into());
        }
        optimizer.backward_step(&current)?;
        iterations = iteration;
        if iteration % 25 == 0 || value < 0.01 {
            println!("iteration {iteration:3} loss {value:.6}");
        }
        if value < 0.01 {
            break;
        }
    }
    let final_loss = mean_loss(&mut model, &examples, &root_step, &targets)?.to_scalar::<f32>()?;
    let predictions = [
        prediction(&mut model, &examples[0], &root_step)?,
        prediction(&mut model, &examples[1], &root_step)?,
    ];
    println!("model: {MODEL_ID}@{REVISION}");
    println!("device: {} dtype: {dtype:?}", device_name(&device));
    println!(
        "hidden size: {} trainable variables: {trainable_count}",
        model.hidden_size()
    );
    println!("iterations: {iterations}");
    println!("initial loss: {initial_loss:.6} final loss: {final_loss:.6}");
    println!("predictions: {predictions:?}");
    if final_loss >= initial_loss * 0.5 || predictions != [0, 1] {
        return Err("structural overfit acceptance criteria failed".into());
    }
    Ok(())
}

fn inference_sequence(
    tokenizer: &Tokenizer,
    prompt: &str,
    identity: &TokenizerIdentity,
    step: &ModelStep,
) -> Result<CausalSequence, Box<dyn std::error::Error>> {
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|error| format!("cannot tokenize {prompt:?}: {error}"))?;
    let position = StructuralPosition::from_model_step(step, PreviousAction::Bos, 0)
        .map_err(|error| format!("cannot make structural position: {error:?}"))?;
    Ok(CausalSequence::from_inference(
        encoding.get_ids(),
        identity,
        &[],
        position,
    ))
}

fn mean_loss(
    model: &mut PretrainedStructuralModel,
    examples: &[CausalSequence],
    step: &ModelStep,
    targets: &[Tensor; 2],
) -> candle_core::Result<Tensor> {
    let first = loss::cross_entropy(&model.fixed_logits(&examples[0], step)?, &targets[0])?;
    let second = loss::cross_entropy(&model.fixed_logits(&examples[1], step)?, &targets[1])?;
    (first + second)? / 2f64
}

fn prediction(
    model: &mut PretrainedStructuralModel,
    sequence: &CausalSequence,
    step: &ModelStep,
) -> candle_core::Result<u32> {
    Ok(model
        .fixed_logits(sequence, step)?
        .argmax(D::Minus1)?
        .to_vec1::<u32>()?[0])
}
