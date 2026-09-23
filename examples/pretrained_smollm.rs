use candle_core::{D, DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig};
use hf_hub::{Repo, RepoType, api::sync::Api};
use mopappa::engine::device::{device_name, selected_device};
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

    let dtype = DType::BF16;
    // The immutable safetensors file remains alive for the duration of model use.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)? };
    let model = Llama::load(vb, &config)?;
    let mut cache = Cache::new(false, dtype, &config, &device)?;
    let input = Tensor::new(ids, &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0, &mut cache)?;
    let next_id = logits.argmax(D::Minus1)?.to_vec1::<u32>()?[0];
    let next = tokenizer
        .decode(&[next_id], true)
        .map_err(|error| format!("cannot decode token: {error}"))?;

    println!("Prompt tokens: {}", ids.len());
    println!("Next token: {next_id} {next:?}");
    Ok(())
}
