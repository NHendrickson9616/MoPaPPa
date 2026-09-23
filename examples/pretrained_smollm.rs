use candle_core::{D, DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig};
use hf_hub::{Repo, RepoType, api::sync::Api};
use mopappa::engine::device::{device_name, selected_device};
use mopappa::model::pretrained_llama::{Cache as LocalCache, Llama as LocalLlama};
use tokenizers::Tokenizer;

const MODEL_ID: &str = "HuggingFaceTB/SmolLM2-135M";
// The checkpoint model card declares Apache-2.0.
const REVISION: &str = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = selected_device()?;
    println!("Device: {}", device_name(&device));
    println!("Model: {MODEL_ID}@{REVISION}");

    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        MODEL_ID.to_owned(),
        RepoType::Model,
        REVISION.to_owned(),
    ));
    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;
    let weights_path = repo.get("model.safetensors")?;

    let llama_config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
    let config = llama_config.into_config(false);
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|error| format!("cannot load tokenizer: {error}"))?;
    let prompt = "Rust is a programming language";
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|error| format!("cannot tokenize prompt: {error}"))?;
    let ids = encoding.get_ids();

    // Candle's CPU matmul does not support BF16; the requested CUDA path does.
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    // The immutable safetensors file remains alive for the duration of model use.
    let input = Tensor::new(ids, &device)?.unsqueeze(0)?;
    let upstream_logits = {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&weights_path),
                dtype,
                &device,
            )?
        };
        let model = Llama::load(vb, &config)?;
        let mut cache = Cache::new(false, dtype, &config, &device)?;
        model.forward(&input, 0, &mut cache)?
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)? };
    let model = LocalLlama::load(vb, &config)?;
    let mut cache = LocalCache::new(false, dtype, &config, &device)?;
    let hidden = model.forward_hidden(&model.embed(&input)?, 0, &mut cache)?;
    assert_eq!(hidden.dims(), &[1, ids.len(), config.hidden_size]);
    let local_logits =
        model.free_token_logits(&hidden.i((.., ids.len() - 1, ..))?.contiguous()?)?;
    let upstream_id = upstream_logits.argmax(D::Minus1)?.to_vec1::<u32>()?[0];
    let next_id = local_logits.argmax(D::Minus1)?.to_vec1::<u32>()?[0];
    assert_eq!(next_id, upstream_id, "local and upstream top tokens differ");
    let max_diff = (&local_logits - &upstream_logits)?
        .abs()?
        .max(D::Minus1)?
        .to_vec1::<f32>()?[0];
    let next = tokenizer
        .decode(&[next_id], true)
        .map_err(|error| format!("cannot decode token: {error}"))?;

    println!("Prompt tokens: {}", ids.len());
    println!("Hidden shape: {:?}", hidden.dims());
    println!("Maximum absolute logits difference: {max_diff:.6}");
    println!("Next token: {next_id} {next:?}");
    Ok(())
}
