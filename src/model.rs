use crate::{config::ModelConfig, preprocess::Window};
use anyhow::{Result, ensure};
use burn::{
    module::{Module, Param},
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
        LinearConfig,
    },
    prelude::*,
    tensor::{Bool, Int, activation, module},
};

#[derive(Clone, Copy, Debug)]
pub struct ExecutionOptions {
    pub folded_embeddings: bool,
    pub fused_qkv: bool,
    pub fused_attention: bool,
    pub flattened_linears: bool,
}

impl Default for ExecutionOptions {
    fn default() -> Self {
        Self {
            folded_embeddings: true,
            ..Self::REFERENCE
        }
    }
}

impl ExecutionOptions {
    pub const REFERENCE: Self = Self {
        folded_embeddings: false,
        fused_qkv: false,
        fused_attention: false,
        flattened_linears: false,
    };
}

fn linear(input: Tensor<3>, layer: &Linear, flatten: bool) -> Tensor<3> {
    if flatten {
        let [b, n, h] = input.dims();
        let out = layer.weight.val().dims()[1];
        layer
            .forward(input.reshape([b * n, h]))
            .reshape([b, n, out])
    } else {
        layer.forward(input)
    }
}

#[derive(Module, Debug)]
pub struct EncoderBlock {
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
    pub output: Linear,
    pub expand: Linear,
    pub contract: Linear,
    pub attention_alpha: Param<Tensor<1>>,
    pub ffn_alpha: Param<Tensor<1>>,
    pub attention_norm: LayerNorm,
    pub ffn_norm: LayerNorm,
    attention_dropout: Dropout,
    relu_dropout: Dropout,
    residual_dropout: Dropout,
    #[module(skip)]
    heads: usize,
    #[module(skip)]
    rezero: bool,
}

impl EncoderBlock {
    fn new(c: &ModelConfig, device: &Device) -> Self {
        let linear = || {
            LinearConfig::new(c.hidden_size, c.hidden_size)
                .with_bias(false)
                .init(device)
        };
        Self {
            query: linear(),
            key: linear(),
            value: linear(),
            output: linear(),
            expand: LinearConfig::new(c.hidden_size, c.filter_size).init(device),
            contract: LinearConfig::new(c.filter_size, c.hidden_size).init(device),
            attention_alpha: Param::from_tensor(Tensor::zeros([1], device)),
            ffn_alpha: Param::from_tensor(Tensor::zeros([1], device)),
            attention_norm: LayerNormConfig::new(c.hidden_size)
                .with_epsilon(1e-6)
                .init(device),
            ffn_norm: LayerNormConfig::new(c.hidden_size)
                .with_epsilon(1e-6)
                .init(device),
            attention_dropout: DropoutConfig::new(c.attention_dropout).init(),
            relu_dropout: DropoutConfig::new(c.relu_dropout).init(),
            residual_dropout: DropoutConfig::new(c.layer_postprocess_dropout).init(),
            heads: c.num_heads,
            rezero: c.rezero,
        }
    }
    fn forward(
        &self,
        x: Tensor<3>,
        mask: Tensor<4, Bool>,
        training: bool,
        options: ExecutionOptions,
    ) -> Tensor<3> {
        let [b, n, h] = x.dims();
        let d = h / self.heads;
        let a = if self.rezero {
            x.clone()
        } else {
            self.attention_norm.forward(x.clone())
        };
        let (q, k, v) = if options.fused_qkv {
            // Keep the original parameters and their gradients/checkpoint names.
            let weight = Tensor::cat(
                vec![
                    self.query.weight.val(),
                    self.key.weight.val(),
                    self.value.weight.val(),
                ],
                1,
            );
            let qkv = if options.flattened_linears {
                module::linear(a.reshape([b * n, h]), weight, None).reshape([b, n, 3 * h])
            } else {
                module::linear(a, weight, None)
            };
            (
                qkv.clone().slice([0..b, 0..n, 0..h]),
                qkv.clone().slice([0..b, 0..n, h..2 * h]),
                qkv.slice([0..b, 0..n, 2 * h..3 * h]),
            )
        } else {
            (
                linear(a.clone(), &self.query, options.flattened_linears),
                linear(a.clone(), &self.key, options.flattened_linears),
                linear(a, &self.value, options.flattened_linears),
            )
        };
        let q = q.reshape([b, n, self.heads, d]).permute([0, 2, 1, 3]) / (d as f32).sqrt();
        let k = k.reshape([b, n, self.heads, d]).permute([0, 2, 1, 3]);
        let v = v.reshape([b, n, self.heads, d]).permute([0, 2, 1, 3]);
        let a = if options.fused_attention && !training {
            module::attention(
                q,
                k,
                v,
                Some(mask),
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(1.0),
                    ..Default::default()
                },
            )
        } else {
            let scores = q.matmul(k.swap_dims(2, 3)).mask_fill(mask, -1e9);
            let p = activation::softmax(scores, 3);
            let p = if training {
                self.attention_dropout.forward(p)
            } else {
                p
            };
            p.matmul(v)
        };
        let a = a.permute([0, 2, 1, 3]).reshape([b, n, h]);
        let a = linear(a, &self.output, options.flattened_linears);
        let a = if training {
            self.residual_dropout.forward(a)
        } else {
            a
        };
        let x = if self.rezero {
            x + a * self.attention_alpha.val().reshape([1, 1, 1])
        } else {
            x + a
        };
        let y = if self.rezero {
            x.clone()
        } else {
            self.ffn_norm.forward(x.clone())
        };
        let y = activation::relu(linear(y, &self.expand, options.flattened_linears));
        let y = if training {
            self.relu_dropout.forward(y)
        } else {
            y
        };
        let y = linear(y, &self.contract, options.flattened_linears);
        let y = if training {
            self.residual_dropout.forward(y)
        } else {
            y
        };
        if self.rezero {
            x + y * self.ffn_alpha.val().reshape([1, 1, 1])
        } else {
            x + y
        }
    }
}

#[derive(Module, Debug)]
pub struct DeepConsensus {
    pub bases: Embedding,
    pub pw: Embedding,
    pub ip: Embedding,
    pub strand: Embedding,
    pub bq: Embedding,
    pub sn: Embedding,
    pub condenser: Linear,
    pub blocks: Vec<EncoderBlock>,
    pub final_norm: LayerNorm,
    pub classifier: Linear,
    input_dropout: Dropout,
    #[module(skip)]
    pub config: ModelConfig,
    #[module(skip)]
    position: Tensor<3>,
    #[module(skip)]
    mask: Tensor<4, Bool>,
}

pub fn positional_encoding(n: usize, h: usize) -> Vec<f32> {
    let half = h / 2;
    let mut values = vec![0.0; n * h];
    for i in 0..n {
        for k in 0..half {
            let angle = i as f64 * (-(k as f64) * 10000f64.ln() / (half - 1) as f64).exp();
            values[i * h + k] = angle.sin() as f32;
            values[i * h + half + k] = angle.cos() as f32;
        }
    }
    values
}

impl DeepConsensus {
    pub fn new(c: &ModelConfig, device: &Device) -> Result<Self> {
        c.validate()?;
        let embedding = |v, h| EmbeddingConfig::new(v, h).init(device);
        let mut mask = Vec::with_capacity(c.max_length * c.max_length);
        for i in 0..c.max_length {
            for j in 0..c.max_length {
                mask.push(
                    c.attn_win_size
                        .filter(|&w| w > 0)
                        .is_some_and(|w| i.abs_diff(j) > w),
                );
            }
        }
        Ok(Self {
            bases: embedding(5, c.per_base_hidden_size),
            pw: embedding(c.pw_max + 1, c.pw_hidden_size),
            ip: embedding(c.ip_max + 1, c.ip_hidden_size),
            strand: embedding(3, c.strand_hidden_size),
            bq: embedding(c.ccs_bq_max, c.ccs_bq_hidden_size),
            sn: embedding(c.sn_max + 1, c.sn_hidden_size),
            condenser: LinearConfig::new(c.embedding_width(), c.hidden_size)
                .with_bias(false)
                .init(device),
            blocks: (0..c.num_hidden_layers)
                .map(|_| EncoderBlock::new(c, device))
                .collect(),
            final_norm: LayerNormConfig::new(c.hidden_size)
                .with_epsilon(1e-6)
                .init(device),
            classifier: LinearConfig::new(c.hidden_size, 5).init(device),
            input_dropout: DropoutConfig::new(c.layer_postprocess_dropout).init(),
            config: c.clone(),
            position: Tensor::from_data(
                TensorData::new(
                    positional_encoding(c.max_length, c.hidden_size),
                    [1, c.max_length, c.hidden_size],
                ),
                device,
            ),
            mask: Tensor::from_data(
                TensorData::new(mask, [1, 1, c.max_length, c.max_length]),
                device,
            ),
        })
    }
    fn embed(
        rows: Tensor<3>,
        from: usize,
        count: usize,
        embedding: &Embedding,
        shift: f32,
        max: usize,
        folded: bool,
    ) -> Tensor<3> {
        let [b, _, n] = rows.dims();
        let width = embedding.weight.val().dims()[1];
        let ids: Tensor<2, Int> = (rows
            .slice([0..b, from..from + count, 0..n])
            .permute([0, 2, 1])
            + shift)
            .clamp(0.0, max as f32)
            .int()
            .reshape([b, n * count]);
        let values = if folded {
            let weight = (embedding.weight.val() * (width as f32).sqrt()).slice_assign(
                [0..1, 0..width],
                Tensor::zeros([1, width], &embedding.weight.val().device()),
            );
            module::embedding(weight, ids)
        } else {
            let zero = ids.clone().equal_elem(0).reshape([b, n * count, 1]);
            (embedding.forward(ids) * (width as f32).sqrt()).mask_fill(zero, 0.0)
        };
        values.reshape([b, n, count * width])
    }
    pub fn forward(&self, rows: Tensor<3>, training: bool) -> Tensor<3> {
        self.forward_with_options(rows, training, ExecutionOptions::default())
    }
    pub fn forward_with_options(
        &self,
        rows: Tensor<3>,
        training: bool,
        options: ExecutionOptions,
    ) -> Tensor<3> {
        let x = self.encode_with_options(rows, training, options);
        self.forward_encoded_with_options(x, training, options)
    }
    pub fn encode_with_options(
        &self,
        rows: Tensor<3>,
        training: bool,
        options: ExecutionOptions,
    ) -> Tensor<3> {
        let c = &self.config;
        let p = c.max_passes;
        // Grouped lookups replace the upstream per-row embedding dispatches.
        let mut features = vec![
            Self::embed(
                rows.clone(),
                0,
                p,
                &self.bases,
                0.0,
                4,
                options.folded_embeddings,
            ),
            Self::embed(
                rows.clone(),
                p,
                p,
                &self.pw,
                0.0,
                c.pw_max,
                options.folded_embeddings,
            ),
            Self::embed(
                rows.clone(),
                2 * p,
                p,
                &self.ip,
                0.0,
                c.ip_max,
                options.folded_embeddings,
            ),
            Self::embed(
                rows.clone(),
                3 * p,
                p,
                &self.strand,
                0.0,
                2,
                options.folded_embeddings,
            ),
            Self::embed(
                rows.clone(),
                4 * p,
                1,
                &self.bases,
                0.0,
                4,
                options.folded_embeddings,
            ),
        ];
        if c.use_ccs_bq {
            features.push(Self::embed(
                rows.clone(),
                4 * p + 1,
                1,
                &self.bq,
                1.0,
                c.ccs_bq_max - 1,
                options.folded_embeddings,
            ));
        }
        features.push(Self::embed(
            rows,
            c.rows() - 4,
            4,
            &self.sn,
            0.0,
            c.sn_max,
            options.folded_embeddings,
        ));
        let mut x = linear(
            Tensor::cat(features, 2),
            &self.condenser,
            options.flattened_linears,
        );
        if c.add_pos_encoding {
            x = x + self.position.clone();
        }
        if training {
            x = self.input_dropout.forward(x);
        }
        x
    }
    pub fn forward_encoded_with_options(
        &self,
        mut x: Tensor<3>,
        training: bool,
        options: ExecutionOptions,
    ) -> Tensor<3> {
        for block in &self.blocks {
            x = block.forward(x, self.mask.clone(), training, options);
        }
        linear(
            self.final_norm.forward(x),
            &self.classifier,
            options.flattened_linears,
        )
    }
}

pub fn batch(windows: &[Window], c: &ModelConfig, device: &Device) -> Result<Tensor<3>> {
    ensure!(!windows.is_empty(), "empty batch");
    let mut data = Vec::with_capacity(windows.len() * c.rows() * c.max_length);
    for w in windows {
        ensure!(
            !w.overflow && w.length == c.max_length && w.rows.len() == c.rows() * c.max_length,
            "window dimensions incompatible with model"
        );
        ensure!(w.rows.iter().all(|x| x.is_finite()), "non-finite feature");
        data.extend_from_slice(&w.rows);
    }
    Ok(Tensor::from_data(
        TensorData::new(data, [windows.len(), c.rows(), c.max_length]),
        device,
    ))
}

pub fn device(name: &str) -> Result<Device> {
    if name == "cpu" {
        return Ok(burn::backend::FlexDevice.into());
    }
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    if let Some(index) = name.strip_prefix("cuda:") {
        return Ok(burn::backend::CudaDevice::new(index.parse()?).into());
    }
    if name == "metal" {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            static METAL: std::sync::OnceLock<std::result::Result<Device, String>> =
                std::sync::OnceLock::new();
            return METAL
                .get_or_init(|| {
                    Device::wgpu_options()
                        .graphics_api(burn::tensor::wgpu::WgpuBackend::Metal)
                        .init()
                        .map_err(|error| error.to_string())
                })
                .as_ref()
                .cloned()
                .map_err(|error| anyhow::anyhow!("Metal initialization failed: {error}"));
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        anyhow::bail!("Metal requires macOS and --features metal (use --device metal)");
    }
    anyhow::bail!(
        "device {name} unavailable; use cpu, cuda:N (Linux --features cuda), or metal (macOS --features metal)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tf_sin_cos_layout() {
        let x = positional_encoding(2, 8);
        assert_eq!(&x[..4], &[0.0; 4]);
        assert_eq!(&x[4..8], &[1.0; 4]);
        assert!((x[8] - 1f32.sin()).abs() < 1e-7);
    }
    #[test]
    fn tiny_forward_backward() {
        let d = Device::autodiff(device("cpu").unwrap());
        let c = ModelConfig {
            max_passes: 1,
            max_length: 4,
            hidden_size: 8,
            num_heads: 2,
            num_hidden_layers: 1,
            filter_size: 16,
            ..Default::default()
        };
        let m = DeepConsensus::new(&c, &d).unwrap();
        let x = Tensor::zeros([1, c.rows(), 4], &d);
        let y = m.forward(x, true);
        assert_eq!(y.dims(), [1, 4, 5]);
        let grads = y.sum().backward();
        assert!(m.classifier.weight.val().grad(&grads).is_some());
    }

    #[test]
    fn optimized_forward_and_gradients_match() {
        let d = Device::autodiff(device("cpu").unwrap());
        d.seed(7);
        let c = ModelConfig {
            max_passes: 2,
            max_length: 7,
            hidden_size: 8,
            num_heads: 2,
            num_hidden_layers: 2,
            filter_size: 16,
            attn_win_size: Some(2),
            attention_dropout: 0.0,
            ..Default::default()
        };
        let mut m = DeepConsensus::new(&c, &d).unwrap();
        for block in &mut m.blocks {
            block.attention_alpha = Param::from_tensor(Tensor::full([1], 0.3, &d));
            block.ffn_alpha = Param::from_tensor(Tensor::full([1], 0.4, &d));
        }
        let values: Vec<f32> = (0..2 * c.rows() * c.max_length)
            .map(|i| (i % 11) as f32 - 2.5)
            .collect();
        let rows = Tensor::from_data(TensorData::new(values, [2, c.rows(), c.max_length]), &d);
        let optimized = ExecutionOptions {
            folded_embeddings: true,
            fused_qkv: true,
            fused_attention: true,
            flattened_linears: true,
        };
        let close = |a: Tensor<3>, b: Tensor<3>| {
            let a = a.to_data().try_to_vec::<f32>().unwrap();
            let b = b.to_data().try_to_vec::<f32>().unwrap();
            assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-5));
        };
        let a = m.forward_with_options(rows.clone(), false, ExecutionOptions::REFERENCE);
        let b = m.forward_with_options(rows.clone(), false, optimized);
        close(a, b);
        let a = m.forward_with_options(rows.clone(), true, ExecutionOptions::REFERENCE);
        let b = m.forward_with_options(rows, true, optimized);
        close(a.clone(), b.clone());
        let ga = a.powf_scalar(2.0).mean().backward();
        let gb = b.powf_scalar(2.0).mean().backward();
        for param in [
            &m.classifier.weight,
            &m.blocks[0].query.weight,
            &m.bases.weight,
            &m.bq.weight,
        ] {
            let a = param
                .val()
                .grad(&ga)
                .unwrap()
                .to_data()
                .try_to_vec::<f32>()
                .unwrap();
            let b = param
                .val()
                .grad(&gb)
                .unwrap()
                .to_data()
                .try_to_vec::<f32>()
                .unwrap();
            assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-5));
        }
    }
}
