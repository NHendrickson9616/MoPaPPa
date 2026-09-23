use super::*;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

fn config() -> DecoderConfig {
    DecoderConfig {
        vocab_size: 17,
        d_model: 8,
        num_heads: 2,
        num_layers: 2,
        d_ff: 16,
        max_seq_len: 8,
        norm_eps: 1e-5,
    }
}

fn decoder<'a>(vars: &'a VarMap, device: &'a Device) -> Decoder {
    Decoder::new(config(), VarBuilder::from_varmap(vars, DType::F32, device)).unwrap()
}

#[test]
fn config_validation_rejects_invalid_values() {
    let mut invalid = config();
    invalid.d_model = 7;
    assert!(invalid.validate().is_err());
    invalid = config();
    invalid.max_seq_len = 0;
    assert!(invalid.validate().is_err());
    invalid = config();
    invalid.norm_eps = f64::NAN;
    assert!(invalid.validate().is_err());
}

#[test]
fn output_shapes_and_no_future_leakage() -> Result<()> {
    let device = Device::Cpu;
    let vars = VarMap::new();
    let model = decoder(&vars, &device);
    let values: Vec<f32> = (0..32).map(|n| n as f32 / 32.0).collect();
    let input = Tensor::from_vec(values, (1, 4, 8), &device)?;
    assert_eq!(model.forward(&input)?.dims(), &[1, 4, 8]);
    assert_eq!(model.last_hidden(&input)?.dims(), &[1, 8]);

    let changed = input.slice_assign(
        &[0..1, 3..4, 0..8],
        &Tensor::full(100f32, (1, 1, 8), &device)?,
    )?;
    let before = model
        .forward(&input)?
        .narrow(1, 0, 3)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let after = model
        .forward(&changed)?
        .narrow(1, 0, 3)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert!(before.iter().zip(after).all(|(a, b)| (a - b).abs() < 1e-5));
    Ok(())
}

#[test]
fn routed_heads_match_every_fixed_training_head() -> Result<()> {
    let device = Device::Cpu;
    let vars = VarMap::new();
    let heads = RoutedHeads::new(8, VarBuilder::from_varmap(&vars, DType::F32, &device))?;
    let hidden = Tensor::zeros((2, 8), DType::F32, &device)?;
    for head in [
        OutputHead::Root,
        OutputHead::ItemList,
        OutputHead::DeclarationKind,
        OutputHead::ParameterList,
        OutputHead::Type,
        OutputHead::Block,
        OutputHead::TypeAnnotation,
        OutputHead::Expression,
        OutputHead::LiteralKind,
        OutputHead::BinaryOperator,
        OutputHead::CallArgument,
        OutputHead::IfElse,
    ] {
        assert_eq!(
            heads.forward(head, &hidden)?.dims(),
            &[2, head.fixed_candidate_count().unwrap()]
        );
    }
    assert!(heads.forward(OutputHead::SymbolPointer, &hidden).is_err());
    assert!(
        heads
            .forward(OutputHead::DirectCallTarget, &hidden)
            .is_err()
    );

    let free_token =
        FreeTokenHead::new(8, 11, VarBuilder::from_varmap(&vars, DType::F32, &device))?;
    assert_eq!(free_token.forward(&hidden)?.dims(), &[2, 11]);
    Ok(())
}

#[test]
fn routed_heads_reject_empty_dimensions() {
    let device = Device::Cpu;
    let vars = VarMap::new();
    let builder = || VarBuilder::from_varmap(&vars, DType::F32, &device);
    assert!(RoutedHeads::new(0, builder()).is_err());
    assert!(FreeTokenHead::new(0, 11, builder()).is_err());
    assert!(FreeTokenHead::new(8, 0, builder()).is_err());
}

#[test]
fn legal_mask_rejects_bad_masks_and_hides_candidates() -> Result<()> {
    let device = Device::Cpu;
    let logits = Tensor::new(&[[1f32, 9., 3.], [7., 2., 1.]], &device)?;
    let legal = Tensor::new(&[[1u8, 0, 1], [0, 1, 1]], &device)?;
    let masked = apply_legal_mask(&logits, &legal)?.to_vec2::<f32>()?;
    assert!(masked[0][1].is_infinite() && masked[0][1].is_sign_negative());
    assert_eq!(masked[1][1], 2.);
    assert!(apply_legal_mask(&logits, &Tensor::zeros((2, 2), DType::U8, &device)?).is_err());
    assert!(apply_legal_mask(&logits, &Tensor::zeros((2, 3), DType::U8, &device)?).is_err());
    assert!(apply_legal_mask(&logits, &legal.to_dtype(DType::I64)?).is_err());
    assert!(apply_legal_mask(&logits.to_dtype(DType::U8)?, &legal).is_err());
    assert!(apply_legal_mask(&logits.to_dtype(DType::I64)?, &legal).is_err());
    let f64_masked = apply_legal_mask(&logits.to_dtype(DType::F64)?, &legal)?.to_vec2::<f64>()?;
    assert!(f64_masked[0][1].is_infinite() && f64_masked[0][1].is_sign_negative());
    let non_binary = Tensor::new(&[[1u8, 2, 1], [1, 1, 1]], &device)?;
    assert!(apply_legal_mask(&logits, &non_binary).is_err());
    Ok(())
}

#[test]
fn pointer_scores_dynamic_symbols_and_mask() -> Result<()> {
    let device = Device::Cpu;
    let query = Tensor::new(&[[1f32, 0.], [0., 1.]], &device)?;
    let keys = Tensor::new(
        &[
            [[1f32, 0.], [2., 0.], [3., 0.]],
            [[0., 1.], [0., 2.], [0., 3.]],
        ],
        &device,
    )?;
    let mask = Tensor::new(&[[1u8, 0, 1], [1, 1, 0]], &device)?;
    let scores = symbol_pointer_scores(&query, &keys, &mask)?.to_vec2::<f32>()?;
    assert_eq!(scores[0][0], 1.);
    assert!(scores[0][1].is_infinite());
    assert_eq!(scores[1][1], 2.);
    assert_eq!(symbol_pointer_scores(&query, &keys, &mask)?.dims(), &[2, 3]);
    Ok(())
}

#[test]
fn parameter_group_contract_is_stable() {
    assert_eq!(
        parameter_groups(),
        [
            ("backbone", "backbone"),
            ("pretrained", "pretrained"),
            ("new_structural", "structural")
        ]
    );
}

#[test]
fn actual_parameter_names_belong_to_exactly_one_group() -> Result<()> {
    let device = Device::Cpu;
    let vars = VarMap::new();
    let _decoder = decoder(&vars, &device);
    let _heads = RoutedHeads::new(8, VarBuilder::from_varmap(&vars, DType::F32, &device))?;
    let _free_token =
        FreeTokenHead::new(8, 11, VarBuilder::from_varmap(&vars, DType::F32, &device))?;

    // VarMap does not expose names in Candle 0.11, so inspect its serialized
    // safetensors header (without adding a serialization dependency).
    let path = std::env::temp_dir().join(format!(
        "mopappa-parameter-groups-{}-{}.safetensors",
        std::process::id(),
        vars.all_vars().len()
    ));
    vars.save(&path)?;
    let bytes = std::fs::read(&path)?;
    std::fs::remove_file(path)?;
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&bytes[8..8 + header_len]).unwrap();
    let names: Vec<&str> = header
        .split('"')
        .filter(|part| {
            part.starts_with("backbone.")
                || part.starts_with("pretrained.")
                || part.starts_with("structural.")
        })
        .collect();
    assert_eq!(names.len(), vars.all_vars().len());
    for name in names {
        let memberships = parameter_groups()
            .iter()
            .filter(|(_, prefix)| name.starts_with(&format!("{prefix}.")))
            .count();
        assert_eq!(memberships, 1, "{name}");
    }
    Ok(())
}

#[test]
fn fixed_attention_is_causal_across_batches_and_heads() -> Result<()> {
    let device = Device::Cpu;
    let mut vars = VarMap::new();
    let cfg = DecoderConfig {
        d_model: 4,
        num_heads: 2,
        ..config()
    };
    let attention = SelfAttention::new(
        &cfg,
        VarBuilder::from_varmap(&vars, DType::F32, &device).pp("fixture"),
    )?;
    let zero_matrix = Tensor::zeros((4, 4), DType::F32, &device)?;
    let identity = Tensor::eye(4, DType::F32, &device)?;
    let zero_bias = Tensor::zeros(4, DType::F32, &device)?;
    for projection in ["q", "k"] {
        vars.set_one(format!("fixture.{projection}.weight"), &zero_matrix)?;
        vars.set_one(format!("fixture.{projection}.bias"), &zero_bias)?;
    }
    for projection in ["v", "output"] {
        vars.set_one(format!("fixture.{projection}.weight"), &identity)?;
        vars.set_one(format!("fixture.{projection}.bias"), &zero_bias)?;
    }
    let input = Tensor::new(
        &[
            [[1f32, 2., 10., 20.], [3., 4., 30., 40.], [5., 6., 50., 60.]],
            [
                [2., 4., 20., 40.],
                [6., 8., 60., 80.],
                [10., 12., 100., 120.],
            ],
        ],
        &device,
    )?;
    let actual = attention.forward(&input)?.to_vec3::<f32>()?;
    let expected = [
        [[1f32, 2., 10., 20.], [2., 3., 20., 30.], [3., 4., 30., 40.]],
        [[2., 4., 20., 40.], [4., 6., 40., 60.], [6., 8., 60., 80.]],
    ];
    for (actual_batch, expected_batch) in actual.iter().zip(expected) {
        for (actual_row, expected_row) in actual_batch.iter().zip(expected_batch) {
            for (actual, expected) in actual_row.iter().zip(expected_row) {
                assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
            }
        }
    }
    Ok(())
}
