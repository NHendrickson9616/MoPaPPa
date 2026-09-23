use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::llama::LlamaConfig;
use hf_hub::{Repo, RepoType, api::sync::Api};
use mopappa::{
    engine::device::{device_name, selected_device},
    model::{
        bridge::ModelValue, ir::SymbolId, pretrained_structural::PretrainedStructuralModel,
        runner::GenerationRunner, sequence::TokenizerIdentity,
    },
    naming::NameRegistry,
    renderer,
    training::OutputHead,
};
use tokenizers::Tokenizer;

const MODEL_ID: &str = "HuggingFaceTB/SmolLM2-135M";
const REVISION: &str = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2";
const PROMPT: &str = "write a Rust function that adds one to its integer parameter";

fn fixed(
    runner: &mut GenerationRunner<PretrainedStructuralModel>,
    head: OutputHead,
    id: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("{head:?}: {:?}", runner.fixed_logits()?.to_vec2::<f32>()?);
    runner.apply_fixed(head, id)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = selected_device()?;
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
    let encoding = tokenizer
        .encode(PROMPT, true)
        .map_err(|error| format!("cannot tokenize prompt: {error}"))?;
    let identity = TokenizerIdentity::new(MODEL_ID, REVISION, "tokenizer.json")
        .map_err(|error| format!("invalid tokenizer identity: {error:?}"))?;
    let backbone_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)? };
    let structural_vars = VarMap::new();
    let structural_vb = VarBuilder::from_varmap(&structural_vars, dtype, &device);
    let model = PretrainedStructuralModel::new(
        config,
        identity.clone(),
        dtype,
        backbone_vb,
        structural_vb,
    )?;
    let hidden_width = model.hidden_size();
    let mut runner = GenerationRunner::new(model, encoding.get_ids().to_vec(), identity);

    println!("device: {}", device_name(&device));
    println!("hidden width: {hidden_width}");

    // Header first, followed by the exact expression trace `parameter + 1`.
    fixed(&mut runner, OutputHead::Root, 0)?;
    fixed(&mut runner, OutputHead::ItemList, 0)?;
    fixed(&mut runner, OutputHead::DeclarationKind, 0)?;
    fixed(&mut runner, OutputHead::ItemList, 1)?;
    fixed(&mut runner, OutputHead::ParameterList, 0)?;
    fixed(&mut runner, OutputHead::DeclarationKind, 1)?;
    fixed(&mut runner, OutputHead::Type, 2)?;
    fixed(&mut runner, OutputHead::ParameterList, 1)?;
    fixed(&mut runner, OutputHead::Type, 2)?;
    fixed(&mut runner, OutputHead::Block, 2)?;
    fixed(&mut runner, OutputHead::Expression, 2)?;
    fixed(&mut runner, OutputHead::BinaryOperator, 0)?;
    fixed(&mut runner, OutputHead::Expression, 0)?;

    let symbols = runner.pointer_symbols()?;
    let parameter_slot = symbols
        .iter()
        .position(|symbol| *symbol == SymbolId(1))
        .ok_or("parameter SymbolId(1) is not a legal pointer candidate")?;
    let pointer_logits = runner.pointer_logits()?.to_vec2::<f32>()?;
    println!("pointer symbols: {symbols:?}");
    println!("pointer logits: {pointer_logits:?}");
    runner.apply_pointer(parameter_slot)?;

    fixed(&mut runner, OutputHead::Expression, 1)?;
    runner.apply_value(ModelValue::Integer(1))?;

    let root = runner.finish()?;
    let mut names = NameRegistry::new();
    names.register(SymbolId(0))?;
    names.register(SymbolId(1))?;
    names.seal();
    names.assign(SymbolId(0), "increment")?;
    names.assign(SymbolId(1), "value")?;
    let rendered = renderer::render(&root, &names)?;
    let expected = "fn increment(value: i32) -> i32 {\n    value + 1\n}";
    if rendered != expected {
        return Err(format!("unexpected generated source:\n{rendered}").into());
    }
    println!("rendered output:\n{rendered}");
    Ok(())
}
