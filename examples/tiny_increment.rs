use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};
use mopappa::{
    engine::device::{device_name, selected_device},
    model::{
        bridge::ModelValue, decoder::DecoderConfig, embeddings::EmbeddingConfig, ir::SymbolId,
        runner::GenerationRunner, sequence::TokenizerIdentity, structural::StructuralModel,
    },
    naming::NameRegistry,
    renderer,
    training::OutputHead,
};

fn fixed(
    runner: &mut GenerationRunner,
    head: OutputHead,
    id: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("{head:?}: {:?}", runner.fixed_logits()?.to_vec2::<f32>()?);
    runner.apply_fixed(head, id)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = selected_device()?;
    println!("Device: {}", device_name(&device));
    let tokenizer = TokenizerIdentity::new("tiny-increment", "1", "sixteen-tokens")
        .map_err(|error| format!("{error:?}"))?;
    let variables = VarMap::new();
    let model = StructuralModel::new(
        EmbeddingConfig {
            vocab_size: 16,
            d_model: 8,
            max_seq_len: 32,
            tokenizer: tokenizer.clone(),
        },
        DecoderConfig {
            vocab_size: 16,
            d_model: 8,
            num_heads: 2,
            num_layers: 1,
            d_ff: 16,
            max_seq_len: 32,
            norm_eps: 1e-5,
        },
        VarBuilder::from_varmap(&variables, DType::F32, &device),
    )?;
    // In-vocabulary stand-ins for the short English prefix "increment count".
    let mut runner = GenerationRunner::new(model, vec![3, 7], tokenizer);

    fixed(&mut runner, OutputHead::Root, 0)?;
    fixed(&mut runner, OutputHead::ItemList, 0)?;
    fixed(&mut runner, OutputHead::DeclarationKind, 0)?; // function @0
    fixed(&mut runner, OutputHead::ItemList, 1)?;
    fixed(&mut runner, OutputHead::ParameterList, 0)?;
    fixed(&mut runner, OutputHead::DeclarationKind, 1)?; // parameter @1
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
        .ok_or("parameter @1 is not a legal pointer candidate")?;
    println!(
        "SymbolPointer {symbols:?}: {:?}",
        runner.pointer_logits()?.to_vec2::<f32>()?
    );
    runner.apply_pointer(parameter_slot)?;

    fixed(&mut runner, OutputHead::Expression, 1)?;
    runner.apply_value(ModelValue::Integer(1))?;

    let root = runner.finish()?;
    let mut names = NameRegistry::new();
    names.register(SymbolId(0))?;
    names.register(SymbolId(1))?;
    names.seal();
    names.assign(SymbolId(0), "increment")?;
    names.assign(SymbolId(1), "count")?;
    let rendered = renderer::render(&root, &names)?;
    let expected = "fn increment(count: i32) -> i32 {\n    count + 1\n}";
    if rendered != expected {
        return Err(format!("unexpected generated source:\n{rendered}").into());
    }
    println!("{rendered}");
    Ok(())
}
