use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use mopappa::{
    model::{
        decoder::DecoderConfig, embeddings::EmbeddingConfig, runner::GenerationRunner,
        sequence::TokenizerIdentity, structural::StructuralModel,
    },
    naming::NameRegistry,
    renderer,
    training::OutputHead,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let tokenizer = TokenizerIdentity::new("tiny-demo", "1", "eight-tokens")
        .map_err(|error| format!("{error:?}"))?;
    let embeddings = EmbeddingConfig {
        vocab_size: 8,
        d_model: 8,
        max_seq_len: 8,
        tokenizer: tokenizer.clone(),
    };
    let decoder = DecoderConfig {
        vocab_size: 8,
        d_model: 8,
        num_heads: 2,
        num_layers: 1,
        d_ff: 16,
        max_seq_len: 8,
        norm_eps: 1e-5,
    };
    let variables = VarMap::new();
    let model = StructuralModel::new(
        embeddings,
        decoder,
        VarBuilder::from_varmap(&variables, DType::F32, &device),
    )?;
    // Arbitrary in-vocabulary stand-ins for a short English prompt.
    let mut runner = GenerationRunner::new(model, vec![2, 5], tokenizer);

    println!(
        "Root masked logits: {:?}",
        runner.fixed_logits()?.to_vec2::<f32>()?
    );
    runner.apply_fixed(OutputHead::Root, 1)?;

    println!(
        "Block masked logits: {:?}",
        runner.fixed_logits()?.to_vec2::<f32>()?
    );
    runner.apply_fixed(OutputHead::Block, 3)?;

    let root = runner.finish()?;
    let mut names = NameRegistry::new();
    names.seal();
    println!("Rendered: {}", renderer::render(&root, &names)?);
    Ok(())
}
