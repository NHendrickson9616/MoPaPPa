//! A small decoder-only backbone and custom output-head foundations.
//!
//! This phase deliberately starts at already-composed `[batch, sequence,
//! d_model]` embeddings. English/action/Need composition, pretrained
//! checkpoints, KV caching, and optimizer policy belong to later phases.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder, layer_norm, linear};

/// Prefixes are part of the public parameter contract used by future trainers.
pub const BACKBONE_PARAMETER_PREFIX: &str = "backbone";
pub const STRUCTURAL_PARAMETER_PREFIX: &str = "structural";

#[derive(Clone, Debug, PartialEq)]
pub struct DecoderConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub norm_eps: f64,
}

impl DecoderConfig {
    pub fn validate(&self) -> Result<()> {
        let positive = [
            ("vocab_size", self.vocab_size),
            ("d_model", self.d_model),
            ("num_heads", self.num_heads),
            ("num_layers", self.num_layers),
            ("d_ff", self.d_ff),
            ("max_seq_len", self.max_seq_len),
        ];
        for (name, value) in positive {
            if value == 0 {
                candle_core::bail!("{name} must be greater than zero")
            }
        }
        if !self.d_model.is_multiple_of(self.num_heads) {
            candle_core::bail!("d_model must be divisible by num_heads")
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            candle_core::bail!("norm_eps must be finite and greater than zero")
        }
        Ok(())
    }
}

struct SelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    output: Linear,
    heads: usize,
}

impl SelfAttention {
    fn new(config: &DecoderConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            q: linear(config.d_model, config.d_model, vb.pp("q"))?,
            k: linear(config.d_model, config.d_model, vb.pp("k"))?,
            v: linear(config.d_model, config.d_model, vb.pp("v"))?,
            output: linear(config.d_model, config.d_model, vb.pp("output"))?,
            heads: config.num_heads,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, seq, width) = input.dims3()?;
        let head_width = width / self.heads;
        let split = |projection: &Linear| {
            projection
                .forward(input)?
                .reshape((batch, seq, self.heads, head_width))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split(&self.q)?;
        let k = split(&self.k)?;
        let v = split(&self.v)?;
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? / (head_width as f64).sqrt())?;
        let mask = causal_additive_mask(seq, input.dtype(), input.device())?;
        let weights = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&mask)?)?;
        let context = weights
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq, width))?;
        self.output.forward(&context)
    }
}

fn causal_additive_mask(seq: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let values: Vec<f32> = (0..seq)
        .flat_map(|row| {
            (0..seq).map(move |column| if column > row { f32::NEG_INFINITY } else { 0.0 })
        })
        .collect();
    Tensor::from_vec(values, (1, 1, seq, seq), device)?.to_dtype(dtype)
}

struct DecoderBlock {
    attention_norm: LayerNorm,
    attention: SelfAttention,
    mlp_norm: LayerNorm,
    mlp_in: Linear,
    mlp_out: Linear,
}

impl DecoderBlock {
    fn new(config: &DecoderConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            attention_norm: layer_norm(config.d_model, config.norm_eps, vb.pp("attention_norm"))?,
            attention: SelfAttention::new(config, vb.pp("attention"))?,
            mlp_norm: layer_norm(config.d_model, config.norm_eps, vb.pp("mlp_norm"))?,
            mlp_in: linear(config.d_model, config.d_ff, vb.pp("mlp.in"))?,
            mlp_out: linear(config.d_ff, config.d_model, vb.pp("mlp.out"))?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let normalized = self.attention_norm.forward(input)?;
        let hidden = (input + self.attention.forward(&normalized)?)?;
        let normalized = self.mlp_norm.forward(&hidden)?;
        let feed_forward = self
            .mlp_out
            .forward(&self.mlp_in.forward(&normalized)?.gelu()?)?;
        hidden + feed_forward
    }
}

/// Decoder-only transformer consuming a single, shared latent width.
pub struct Decoder {
    config: DecoderConfig,
    blocks: Vec<DecoderBlock>,
    final_norm: LayerNorm,
}

impl Decoder {
    pub fn new(config: DecoderConfig, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let vb = vb.pp(BACKBONE_PARAMETER_PREFIX);
        let blocks = (0..config.num_layers)
            .map(|index| DecoderBlock::new(&config, vb.pp(format!("blocks.{index}"))))
            .collect::<Result<Vec<_>>>()?;
        let final_norm = layer_norm(config.d_model, config.norm_eps, vb.pp("final_norm"))?;
        Ok(Self {
            config,
            blocks,
            final_norm,
        })
    }

    /// Returns every contextual hidden state as `[B, T, D]`.
    pub fn forward(&self, embeddings: &Tensor) -> Result<Tensor> {
        let (_, seq, width) = embeddings.dims3()?;
        if width != self.config.d_model {
            candle_core::bail!(
                "embedding width {width} does not match d_model {}",
                self.config.d_model
            )
        }
        if seq == 0 || seq > self.config.max_seq_len {
            candle_core::bail!("sequence length must be in 1..={}", self.config.max_seq_len)
        }
        let mut hidden = embeddings.clone();
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        self.final_norm.forward(&hidden)
    }

    /// Returns `[B, D]` for the last sequence position.
    pub fn last_hidden(&self, embeddings: &Tensor) -> Result<Tensor> {
        let hidden = self.forward(embeddings)?;
        hidden.narrow(1, hidden.dim(1)? - 1, 1)?.squeeze(1)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeadRoute {
    Item,
    Expression,
    Type,
    BinaryOperator,
    ListContinue,
    NewBinding,
    FreeToken,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoutedHeadConfig {
    pub item: usize,
    pub expression: usize,
    pub ty: usize,
    pub binary_operator: usize,
    pub list_continue: usize,
    pub new_binding: usize,
    pub free_token: usize,
}

impl RoutedHeadConfig {
    pub fn validate(&self, d_model: usize) -> Result<()> {
        if d_model == 0 {
            candle_core::bail!("d_model must be greater than zero")
        }
        for (route, size) in [
            ("item", self.item),
            ("expression", self.expression),
            ("type", self.ty),
            ("binary_operator", self.binary_operator),
            ("list_continue", self.list_continue),
            ("new_binding", self.new_binding),
            ("free_token", self.free_token),
        ] {
            if size == 0 {
                candle_core::bail!("{route} head must have at least one candidate")
            }
        }
        Ok(())
    }
}

/// Fixed structural classifiers. Only the selected projection is evaluated.
pub struct RoutedHeads {
    item: Linear,
    expression: Linear,
    ty: Linear,
    binary_operator: Linear,
    list_continue: Linear,
    new_binding: Linear,
    free_token: Linear,
}

impl RoutedHeads {
    pub fn new(d_model: usize, sizes: &RoutedHeadConfig, vb: VarBuilder<'_>) -> Result<Self> {
        sizes.validate(d_model)?;
        let vb = vb.pp(STRUCTURAL_PARAMETER_PREFIX).pp("heads");
        Ok(Self {
            item: linear(d_model, sizes.item, vb.pp("item"))?,
            expression: linear(d_model, sizes.expression, vb.pp("expression"))?,
            ty: linear(d_model, sizes.ty, vb.pp("type"))?,
            binary_operator: linear(d_model, sizes.binary_operator, vb.pp("binary_operator"))?,
            list_continue: linear(d_model, sizes.list_continue, vb.pp("list_continue"))?,
            new_binding: linear(d_model, sizes.new_binding, vb.pp("new_binding"))?,
            free_token: linear(d_model, sizes.free_token, vb.pp("free_token"))?,
        })
    }

    pub fn forward(&self, route: HeadRoute, hidden: &Tensor) -> Result<Tensor> {
        match route {
            HeadRoute::Item => self.item.forward(hidden),
            HeadRoute::Expression => self.expression.forward(hidden),
            HeadRoute::Type => self.ty.forward(hidden),
            HeadRoute::BinaryOperator => self.binary_operator.forward(hidden),
            HeadRoute::ListContinue => self.list_continue.forward(hidden),
            HeadRoute::NewBinding => self.new_binding.forward(hidden),
            HeadRoute::FreeToken => self.free_token.forward(hidden),
        }
    }
}

/// Replaces illegal logits with negative infinity, rejecting unusable masks.
pub fn apply_legal_mask(logits: &Tensor, legal: &Tensor) -> Result<Tensor> {
    if logits.dims() != legal.dims() {
        candle_core::bail!("legal mask shape must exactly match logits")
    }
    if !matches!(logits.dtype(), DType::F32 | DType::F64) {
        candle_core::bail!("logits must have F32 or F64 dtype")
    }
    if legal.dtype() != DType::U8 {
        candle_core::bail!("legal mask must have U8 dtype")
    }
    let last = *logits
        .dims()
        .last()
        .ok_or_else(|| candle_core::Error::Msg("logits must have a candidate dimension".into()))?;
    if last == 0 {
        candle_core::bail!("candidate dimension must not be empty")
    }
    let rows = legal
        .reshape((legal.elem_count() / last, last))?
        .to_vec2::<u8>()?;
    if rows.iter().flatten().any(|&value| value != 0 && value != 1) {
        candle_core::bail!("legal mask values must be binary")
    }
    if rows.iter().any(|row| row.iter().all(|&value| value == 0)) {
        candle_core::bail!("every candidate row must contain a legal choice")
    }
    let illegal = legal.eq(0u8)?;
    let fill = Tensor::full(f32::NEG_INFINITY, logits.shape(), logits.device())?
        .to_dtype(logits.dtype())?;
    illegal.where_cond(&fill, logits)
}

/// Dot-product pointer scores `[B, S]` over runtime-provided symbol keys.
pub fn symbol_pointer_scores(query: &Tensor, keys: &Tensor, legal: &Tensor) -> Result<Tensor> {
    let (batch, width) = query.dims2()?;
    let (key_batch, symbols, key_width) = keys.dims3()?;
    if (batch, width) != (key_batch, key_width) {
        candle_core::bail!("query and symbol key batch/width dimensions must match")
    }
    let scores = keys
        .matmul(&query.unsqueeze(2)?)?
        .reshape((batch, symbols))?;
    apply_legal_mask(&scores, legal)
}

/// Stable high-level groups; this is metadata, not optimizer freezing.
pub fn parameter_groups() -> [(&'static str, &'static str); 2] {
    [
        ("backbone", BACKBONE_PARAMETER_PREFIX),
        ("new_structural", STRUCTURAL_PARAMETER_PREFIX),
    ]
}

#[cfg(test)]
mod tests;
