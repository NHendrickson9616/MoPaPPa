//! LLaMA inference with access to the full, final hidden-state sequence.
//!
//! Adapted from `candle-transformers/src/models/llama.rs` in
//! candle-transformers 0.11.0, commit 31f35b147389700ed2a178ee66a91c3cc25cc80d.
//! Copyright 2023-2025 Hugging Face. Licensed under MIT OR Apache-2.0; see the
//! candle repository's `LICENSE-MIT` and `LICENSE-APACHE` files.

use std::{collections::HashMap, f32::consts::PI};

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{
    Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_no_bias, rms_norm,
};

pub use candle_transformers::models::llama::Config;
use candle_transformers::models::llama::{Llama3RopeConfig, Llama3RopeType};

/// RoPE tables, causal masks, and optional per-layer key/value state.
#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<(usize, usize), Tensor>,
    use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

impl Cache {
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let base_freqs = || {
            (0..head_dim)
                .step_by(2)
                .map(|i| 1f32 / config.rope_theta.powf(i as f32 / head_dim as f32))
                .collect::<Vec<_>>()
        };
        let freqs = match &config.rope_scaling {
            None
            | Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Default,
                ..
            }) => base_freqs(),
            Some(scaling) => {
                let low_wave =
                    scaling.original_max_position_embeddings as f32 / scaling.low_freq_factor;
                let high_wave =
                    scaling.original_max_position_embeddings as f32 / scaling.high_freq_factor;
                base_freqs()
                    .into_iter()
                    .map(|freq| {
                        let wave = 2. * PI / freq;
                        if wave < high_wave {
                            freq
                        } else if wave > low_wave {
                            freq / scaling.factor
                        } else {
                            let smooth = (scaling.original_max_position_embeddings as f32 / wave
                                - scaling.low_freq_factor)
                                / (scaling.high_freq_factor - scaling.low_freq_factor);
                            (1. - smooth) * freq / scaling.factor + smooth * freq
                        }
                    })
                    .collect()
            }
        };
        let theta = Tensor::new(freqs, device)?;
        let positions = Tensor::arange(0, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?;
        let angles = positions.matmul(&theta.reshape((1, theta.elem_count()))?)?;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; config.num_hidden_layers],
            cos: angles.cos()?.to_dtype(dtype)?,
            sin: angles.sin()?.to_dtype(dtype)?,
            device: device.clone(),
        })
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize) -> Result<Tensor> {
        let key = (seq_len, index_pos + seq_len);
        if let Some(mask) = self.masks.get(&key) {
            return Ok(mask.clone());
        }
        let kv_len = index_pos + seq_len;
        let values = (0..seq_len)
            .flat_map(|q| (0..kv_len).map(move |k| u8::from(k > index_pos + q)))
            .collect::<Vec<_>>();
        let mask = Tensor::from_vec(values, (seq_len, kv_len), &self.device)?;
        self.masks.insert(key, mask.clone());
        Ok(mask)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        Ok(Self {
            q: linear_no_bias(
                config.hidden_size,
                head_dim * config.num_attention_heads,
                vb.pp("q_proj"),
            )?,
            k: linear_no_bias(
                config.hidden_size,
                head_dim * config.num_key_value_heads,
                vb.pp("k_proj"),
            )?,
            v: linear_no_bias(
                config.hidden_size,
                head_dim * config.num_key_value_heads,
                vb.pp("v_proj"),
            )?,
            o: linear_no_bias(
                head_dim * config.num_attention_heads,
                config.hidden_size,
                vb.pp("o_proj"),
            )?,
            heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden) = x.dims3()?;
        let shape = (batch, seq_len, self.heads, self.head_dim);
        let q = self
            .q
            .forward(x)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let kv_shape = (batch, seq_len, self.kv_heads, self.head_dim);
        let k = self
            .k
            .forward(x)?
            .reshape(kv_shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let mut v = self.v.forward(x)?.reshape(kv_shape)?.transpose(1, 2)?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        let q = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
        let mut k = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;
        if cache.use_kv_cache {
            if let Some((old_k, old_v)) = &cache.kvs[layer] {
                k = Tensor::cat(&[old_k, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[old_v, &v], 2)?.contiguous()?;
            }
            cache.kvs[layer] = Some((k.clone(), v.clone()));
        }
        let repeats = self.heads / self.kv_heads;
        let repeat = |x: Tensor| -> Result<Tensor> {
            if repeats == 1 {
                Ok(x)
            } else {
                let (b, h, t, d) = x.dims4()?;
                x.unsqueeze(2)?
                    .expand((b, h, repeats, t, d))?
                    .reshape((b, h * repeats, t, d))
            }
        };
        let k = repeat(k)?;
        let v = repeat(v)?;
        let dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        let scores = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let scores = if seq_len == 1 {
            scores
        } else {
            let mask = cache
                .mask(seq_len, index_pos)?
                .broadcast_as(scores.shape())?;
            let minus_inf =
                Tensor::new(f32::NEG_INFINITY, scores.device())?.broadcast_as(scores.shape())?;
            mask.where_cond(&minus_inf, &scores)?
        };
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let y = probs.matmul(&v.contiguous()?)?.to_dtype(dtype)?;
        self.o
            .forward(&y.transpose(1, 2)?.reshape((batch, seq_len, hidden))?)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        Ok(Self {
            gate: linear_no_bias(
                config.hidden_size,
                config.intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up: linear_no_bias(
                config.hidden_size,
                config.intermediate_size,
                vb.pp("up_proj"),
            )?,
            down: linear_no_bias(
                config.intermediate_size,
                config.hidden_size,
                vb.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.down
            .forward(&(candle_nn::ops::silu(&self.gate.forward(x)?)? * self.up.forward(x)?)?)
    }
}

#[derive(Debug, Clone)]
struct Block {
    input_norm: RmsNorm,
    attention: Attention,
    post_attention_norm: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        Ok(Self {
            input_norm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            attention: Attention::load(vb.pp("self_attn"), config)?,
            post_attention_norm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::load(vb.pp("mlp"), config)?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Llama {
    embeddings: Embedding,
    blocks: Vec<Block>,
    final_norm: RmsNorm,
    lm_head: Linear,
}

impl Llama {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(embeddings.embeddings().clone(), None)
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };
        let blocks = (0..config.num_hidden_layers)
            .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), config))
            .collect::<Result<Vec<_>>>()?;
        let final_norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;
        Ok(Self {
            embeddings,
            blocks,
            final_norm,
            lm_head,
        })
    }

    pub fn embed(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.embeddings.forward(token_ids)
    }

    pub fn forward_hidden(
        &self,
        input_embeddings: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let mut x = input_embeddings.clone();
        for (layer, block) in self.blocks.iter().enumerate() {
            let residual = &x;
            x = (block.attention.forward(
                &block.input_norm.forward(&x)?,
                index_pos,
                layer,
                cache,
            )? + residual)?;
            let residual = &x;
            x = (block.mlp.forward(&block.post_attention_norm.forward(&x)?)? + residual)?;
        }
        self.final_norm.forward(&x)
    }

    pub fn free_token_logits(&self, last_hidden: &Tensor) -> Result<Tensor> {
        self.lm_head.forward(last_hidden)?.to_dtype(DType::F32)
    }

    pub fn forward(
        &self,
        token_ids: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (_, seq_len) = token_ids.dims2()?;
        let hidden = self.forward_hidden(&self.embed(token_ids)?, index_pos, cache)?;
        self.free_token_logits(&hidden.i((.., seq_len - 1, ..))?.contiguous()?)
    }
}
