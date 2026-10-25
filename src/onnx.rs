use crate::model::{DeepConsensus, positional_encoding};
#[cfg(feature = "onnx")]
use crate::preprocess::Window;
use anyhow::Result;
use burn::{
    nn::{Embedding, LayerNorm, Linear},
    prelude::*,
};
use prost::Message;
use std::path::Path;

#[derive(Clone, PartialEq, Message)]
pub struct ModelProto {
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    #[prost(string, tag = "2")]
    pub producer_name: String,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "8")]
    pub opsets: Vec<Opset>,
}
#[derive(Clone, PartialEq, Message)]
pub struct Opset {
    #[prost(string, tag = "1")]
    pub domain: String,
    #[prost(int64, tag = "2")]
    pub version: i64,
}
#[derive(Clone, PartialEq, Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    pub nodes: Vec<Node>,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "5")]
    pub initializers: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    pub inputs: Vec<ValueInfo>,
    #[prost(message, repeated, tag = "12")]
    pub outputs: Vec<ValueInfo>,
}
#[derive(Clone, PartialEq, Message)]
pub struct Node {
    #[prost(string, repeated, tag = "1")]
    pub inputs: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub outputs: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op: String,
    #[prost(message, repeated, tag = "5")]
    pub attributes: Vec<Attribute>,
}
#[derive(Clone, PartialEq, Message)]
pub struct Attribute {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(float, tag = "2")]
    pub f: f32,
    #[prost(int64, tag = "3")]
    pub i: i64,
    #[prost(int64, repeated, tag = "8")]
    pub ints: Vec<i64>,
    #[prost(int32, tag = "20")]
    pub kind: i32,
}
#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    pub dtype: i32,
    #[prost(string, tag = "8")]
    pub name: String,
    #[prost(bytes = "vec", tag = "9")]
    pub raw: Vec<u8>,
}
#[derive(Clone, PartialEq, Message)]
pub struct ValueInfo {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,
}
#[derive(Clone, PartialEq, Message)]
pub struct TypeProto {
    #[prost(message, optional, tag = "1")]
    pub tensor: Option<TensorType>,
}
#[derive(Clone, PartialEq, Message)]
pub struct TensorType {
    #[prost(int32, tag = "1")]
    pub elem_type: i32,
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShape>,
}
#[derive(Clone, PartialEq, Message)]
pub struct TensorShape {
    #[prost(message, repeated, tag = "1")]
    pub dims: Vec<Dimension>,
}
#[derive(Clone, PartialEq, Message)]
pub struct Dimension {
    #[prost(int64, optional, tag = "1")]
    pub value: Option<i64>,
    #[prost(string, optional, tag = "2")]
    pub param: Option<String>,
}

fn int(name: &str, value: i64) -> Attribute {
    Attribute {
        name: name.into(),
        kind: 2,
        i: value,
        ..Default::default()
    }
}
fn ints(name: &str, value: &[i64]) -> Attribute {
    Attribute {
        name: name.into(),
        kind: 7,
        ints: value.to_vec(),
        ..Default::default()
    }
}
fn float(name: &str, value: f32) -> Attribute {
    Attribute {
        name: name.into(),
        kind: 1,
        f: value,
        ..Default::default()
    }
}
fn info(name: &str, shape: &[usize]) -> ValueInfo {
    ValueInfo {
        name: name.into(),
        r#type: Some(TypeProto {
            tensor: Some(TensorType {
                elem_type: 1,
                shape: Some(TensorShape {
                    dims: shape
                        .iter()
                        .enumerate()
                        .map(|(i, &n)| Dimension {
                            value: if i == 0 { None } else { Some(n as i64) },
                            param: if i == 0 { Some("batch".into()) } else { None },
                        })
                        .collect(),
                }),
            }),
        }),
    }
}

struct Builder {
    graph: GraphProto,
    counter: usize,
    optimized: bool,
    window_length: usize,
}
impl Builder {
    fn name(&mut self) -> String {
        let n = format!("v{}", self.counter);
        self.counter += 1;
        n
    }
    fn op(&mut self, op: &str, inputs: &[&str], attributes: Vec<Attribute>) -> String {
        let name = self.name();
        self.graph.nodes.push(Node {
            name: name.clone(),
            op: op.into(),
            inputs: inputs.iter().map(|&s| s.into()).collect(),
            outputs: vec![name.clone()],
            attributes,
        });
        name
    }
    fn constant(&mut self, data: Vec<u8>, shape: &[usize], dtype: i32) -> String {
        let name = self.name();
        self.graph.initializers.push(TensorProto {
            name: name.clone(),
            dims: shape.iter().map(|&x| x as i64).collect(),
            dtype,
            raw: data,
        });
        name
    }
    fn floats(&mut self, data: &[f32], shape: &[usize]) -> String {
        self.constant(
            data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            shape,
            1,
        )
    }
    fn longs(&mut self, data: &[i64]) -> String {
        self.constant(
            data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            &[data.len()],
            7,
        )
    }
    fn tensor<const D: usize>(&mut self, t: Tensor<D>) -> Result<String> {
        let shape = t.dims();
        Ok(self.floats(&t.to_data().try_to_vec::<f32>()?, &shape))
    }
    fn reshape(&mut self, x: &str, shape: &[i64]) -> String {
        let shape = self.longs(shape);
        self.op("Reshape", &[x, &shape], vec![])
    }
    fn transpose(&mut self, x: &str, perm: &[i64]) -> String {
        self.op("Transpose", &[x], vec![ints("perm", perm)])
    }
    fn linear(&mut self, x: &str, layer: &Linear) -> Result<String> {
        self.affine(x, layer, false)
    }
    fn affine(&mut self, x: &str, layer: &Linear, relu: bool) -> Result<String> {
        if self.optimized {
            let [input, output] = layer.weight.val().dims();
            let x = self.reshape(x, &[-1, input as i64]);
            let weight = self.tensor(layer.weight.val())?;
            let mut y = if let Some(bias) = &layer.bias {
                let bias = self.tensor(bias.val())?;
                self.op("Gemm", &[&x, &weight, &bias], vec![])
            } else {
                self.op("MatMul", &[&x, &weight], vec![])
            };
            if relu {
                y = self.op("Relu", &[&y], vec![]);
            }
            return Ok(self.reshape(&y, &[-1, self.window_length as i64, output as i64]));
        }
        let weight = self.tensor(layer.weight.val())?;
        let y = self.op("MatMul", &[x, &weight], vec![]);
        let y = if let Some(b) = &layer.bias {
            let bias = self.tensor(b.val())?;
            self.op("Add", &[&y, &bias], vec![])
        } else {
            y
        };
        Ok(if relu {
            self.op("Relu", &[&y], vec![])
        } else {
            y
        })
    }
    fn norm(&mut self, x: &str, norm: &LayerNorm) -> Result<String> {
        let gamma = self.tensor(norm.gamma.val())?;
        let beta = self.tensor(norm.beta.as_ref().unwrap().val())?;
        Ok(self.op(
            "LayerNormalization",
            &[x, &gamma, &beta],
            vec![int("axis", -1), float("epsilon", 1e-6)],
        ))
    }
    fn embedding(
        &mut self,
        from: usize,
        count: usize,
        embedding: &Embedding,
        shift: f32,
        max: usize,
        n: usize,
    ) -> Result<String> {
        let ids = self.longs(&(from..from + count).map(|x| x as i64).collect::<Vec<_>>());
        let x = self.op("Gather", &["rows", &ids], vec![int("axis", 1)]);
        let mut x = self.transpose(&x, &[0, 2, 1]);
        if shift != 0.0 {
            let s = self.floats(&[shift], &[]);
            x = self.op("Add", &[&x, &s], vec![]);
        }
        let lo = self.floats(&[0.0], &[]);
        let hi = self.floats(&[max as f32], &[]);
        let x = self.op("Clip", &[&x, &lo, &hi], vec![]);
        let x = self.op("Cast", &[&x], vec![int("to", 7)]);
        let [v, width] = embedding.weight.val().dims();
        let mut weights = embedding.weight.val().to_data().try_to_vec::<f32>()?;
        for (i, w) in weights.iter_mut().enumerate() {
            *w = if i < width {
                0.0
            } else {
                *w * (width as f32).sqrt()
            };
        }
        let weight = self.floats(&weights, &[v, width]);
        let x = self.op("Gather", &[&weight, &x], vec![int("axis", 0)]);
        Ok(self.reshape(&x, &[0, n as i64, (count * width) as i64]))
    }
}

pub fn export(model: &DeepConsensus, path: &Path) -> Result<()> {
    export_with_options(model, path, false)
}

pub fn config_path(path: &Path) -> std::path::PathBuf {
    path.with_extension("params.json")
}

pub fn validate_export_path(path: &Path) -> Result<()> {
    anyhow::ensure!(
        path.extension().is_some_and(|e| e == "onnx"),
        "ONNX output path must end in .onnx: {}",
        path.display()
    );
    Ok(())
}

pub fn export_with_options(model: &DeepConsensus, path: &Path, optimized: bool) -> Result<()> {
    validate_export_path(path)?;
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let c = &model.config;
    let n = c.max_length;
    let h = c.hidden_size;
    let p = c.max_passes;
    let mut b = Builder {
        graph: GraphProto {
            name: "deepconsensus-rs".into(),
            inputs: vec![info("rows", &[0, c.rows(), n])],
            outputs: vec![info("logits", &[0, n, 5])],
            ..Default::default()
        },
        counter: 0,
        optimized,
        window_length: n,
    };
    let mut parts = vec![
        b.embedding(0, p, &model.bases, 0.0, 4, n)?,
        b.embedding(p, p, &model.pw, 0.0, c.pw_max, n)?,
        b.embedding(2 * p, p, &model.ip, 0.0, c.ip_max, n)?,
        b.embedding(3 * p, p, &model.strand, 0.0, 2, n)?,
        b.embedding(4 * p, 1, &model.bases, 0.0, 4, n)?,
    ];
    if c.use_ccs_bq {
        parts.push(b.embedding(4 * p + 1, 1, &model.bq, 1.0, c.ccs_bq_max - 1, n)?);
    }
    parts.push(b.embedding(c.rows() - 4, 4, &model.sn, 0.0, c.sn_max, n)?);
    let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
    let x = b.op("Concat", &parts, vec![int("axis", 2)]);
    let mut x = b.linear(&x, &model.condenser)?;
    if c.add_pos_encoding {
        let position = b.floats(&positional_encoding(n, h), &[1, n, h]);
        x = b.op("Add", &[&x, &position], vec![]);
    }
    let mask: Vec<f32> = (0..n)
        .flat_map(|i| {
            (0..n).map(move |j| {
                if c.attn_win_size
                    .filter(|&w| w > 0)
                    .is_some_and(|w| i.abs_diff(j) > w)
                {
                    -1e9
                } else {
                    0.0
                }
            })
        })
        .collect();
    let mask = b.floats(&mask, &[1, 1, n, n]);
    for block in &model.blocks {
        let a = if c.rezero {
            x.clone()
        } else {
            b.norm(&x, &block.attention_norm)?
        };
        let (q, k, v) = if optimized {
            let weight = b.tensor(Tensor::cat(
                vec![
                    block.query.weight.val(),
                    block.key.weight.val(),
                    block.value.weight.val(),
                ],
                1,
            ))?;
            let input = b.reshape(&a, &[-1, h as i64]);
            let qkv = b.op("MatMul", &[&input, &weight], vec![]);
            let qkv = b.reshape(
                &qkv,
                &[
                    -1,
                    n as i64,
                    3,
                    c.num_heads as i64,
                    (h / c.num_heads) as i64,
                ],
            );
            let mut parts = Vec::new();
            for index in 0i64..3 {
                let id = b.constant(index.to_le_bytes().to_vec(), &[], 7);
                parts.push(b.op("Gather", &[&qkv, &id], vec![int("axis", 2)]));
            }
            (parts.remove(0), parts.remove(0), parts.remove(0))
        } else {
            let mut parts = Vec::new();
            for layer in [&block.query, &block.key, &block.value] {
                let projected = b.linear(&a, layer)?;
                parts.push(b.reshape(
                    &projected,
                    &[0, n as i64, c.num_heads as i64, (h / c.num_heads) as i64],
                ));
            }
            (parts.remove(0), parts.remove(0), parts.remove(0))
        };
        let q = b.transpose(&q, &[0, 2, 1, 3]);
        let scale = b.floats(&[1.0 / ((h / c.num_heads) as f32).sqrt()], &[]);
        let q = b.op("Mul", &[&q, &scale], vec![]);
        let k = b.transpose(&k, &[0, 2, 3, 1]);
        let v = b.transpose(&v, &[0, 2, 1, 3]);
        let scores = b.op("MatMul", &[&q, &k], vec![]);
        let scores = b.op("Add", &[&scores, &mask], vec![]);
        let scores = b.op("Softmax", &[&scores], vec![int("axis", -1)]);
        let y = b.op("MatMul", &[&scores, &v], vec![]);
        let y = b.transpose(&y, &[0, 2, 1, 3]);
        let y = b.reshape(&y, &[0, n as i64, h as i64]);
        let mut y = b.linear(&y, &block.output)?;
        if c.rezero {
            let alpha = b.tensor(block.attention_alpha.val())?;
            y = b.op("Mul", &[&y, &alpha], vec![]);
        }
        x = b.op("Add", &[&x, &y], vec![]);
        let a = if c.rezero {
            x.clone()
        } else {
            b.norm(&x, &block.ffn_norm)?
        };
        let y = b.affine(&a, &block.expand, true)?;
        let mut y = b.linear(&y, &block.contract)?;
        if c.rezero {
            let alpha = b.tensor(block.ffn_alpha.val())?;
            y = b.op("Mul", &[&y, &alpha], vec![]);
        }
        x = b.op("Add", &[&x, &y], vec![]);
    }
    let x = b.norm(&x, &model.final_norm)?;
    let logits = b.linear(&x, &model.classifier)?;
    b.graph.nodes.push(Node {
        inputs: vec![logits],
        outputs: vec!["logits".into()],
        op: "Identity".into(),
        ..Default::default()
    });
    let model = ModelProto {
        ir_version: 9,
        producer_name: "deepconsensus-rs".into(),
        graph: Some(b.graph),
        opsets: vec![Opset {
            domain: String::new(),
            version: 18,
        }],
    };
    std::fs::write(path, model.encode_to_vec())?;
    c.save(config_path(path))?;
    Ok(())
}

#[cfg(feature = "onnx")]
pub struct OrtEngine {
    session: ort::session::Session,
    config: crate::config::ModelConfig,
}
#[cfg(feature = "onnx")]
impl OrtEngine {
    pub fn new(
        path: &Path,
        config: &crate::config::ModelConfig,
        provider: &str,
        threads: usize,
    ) -> Result<Self> {
        let mut builder = ort::session::Session::builder()?
            .with_intra_threads(threads.max(1))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let (provider, index) = provider
            .split_once(':')
            .unwrap_or((provider, if provider == "coreml" { "gpu" } else { "0" }));
        #[cfg(all(feature = "onnx-coreml", target_os = "macos"))]
        if provider == "coreml" {
            let units = match index {
                "cpu" => ort::ep::coreml::ComputeUnits::CPUOnly,
                "gpu" => ort::ep::coreml::ComputeUnits::CPUAndGPU,
                _ => anyhow::bail!("CoreML provider must be coreml:cpu or coreml:gpu"),
            };
            builder = builder.with_execution_providers([
                ort::ep::CoreML::default()
                    .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
                    .with_compute_units(units)
                    .with_static_input_shapes(false)
                    .with_low_precision_accumulation_on_gpu(false)
                    .with_profile_compute_plan(std::env::var("DC_COREML_PROFILE").as_deref() == Ok("1"))
                    .build()
                    .error_on_failure()
            ]).map_err(|e| anyhow::anyhow!("CoreML registration failed; ORT_DYLIB_PATH must point to a macOS runtime built with CoreML: {e}"))?;
            eprintln!(
                "CoreML selected ({units:?}); unsupported graph partitions may execute on CPU"
            );
            return Self::from_builder(builder, path, config);
        }
        if provider == "coreml" {
            anyhow::bail!("CoreML requires macOS --features metal or onnx-coreml");
        }
        let device_id: i32 = index
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid ORT GPU index: {index}"))?;
        anyhow::ensure!(device_id >= 0, "ORT GPU index must be nonnegative");
        let _ = device_id;
        match provider {
            "cpu" => {
                let env = ort::environment::Environment::current()?;
                let cpus: Vec<_> = env
                    .devices()
                    .filter(|d| d.ep().is_ok_and(|name| name == "CPUExecutionProvider"))
                    .collect();
                anyhow::ensure!(!cpus.is_empty(), "ONNX Runtime exposes no CPU device");
                builder = builder
                    .with_devices(cpus, None)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            #[cfg(all(feature = "tensorrt", target_os = "linux"))]
            "tensorrt" => {
                builder = builder
                    .with_execution_providers([
                        ort::ep::TensorRT::default()
                            .with_device_id(device_id)
                            .with_fp16(std::env::var("DC_TRT_FP16").as_deref() == Ok("1"))
                            .with_engine_cache(true)
                            .with_engine_cache_path(".trt-cache")
                            .build()
                            .error_on_failure(),
                        ort::ep::CUDA::default()
                            .with_device_id(device_id)
                            .build()
                            .error_on_failure(),
                    ])
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            #[cfg(all(feature = "onnx-cuda", target_os = "linux"))]
            "cuda" => {
                builder = builder
                    .with_execution_providers([ort::ep::CUDA::default()
                        .with_device_id(device_id)
                        .build()
                        .error_on_failure()])
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            _ => anyhow::bail!(
                "ORT provider {provider} not compiled; CoreML requires macOS --features metal/onnx-coreml, CUDA requires Linux --features cuda/onnx-cuda, TensorRT requires Linux --features tensorrt"
            ),
        }
        Self::from_builder(builder, path, config)
    }

    fn from_builder(
        mut builder: ort::session::builder::SessionBuilder,
        path: &Path,
        config: &crate::config::ModelConfig,
    ) -> Result<Self> {
        let session = builder.commit_from_file(path)?;
        anyhow::ensure!(
            session.inputs().len() == 1 && session.inputs()[0].name() == "rows",
            "expected deepconsensus-rs ONNX input rows"
        );
        Ok(Self {
            session,
            config: config.clone(),
        })
    }
    pub fn predict(&mut self, windows: &[Window]) -> Result<Vec<f32>> {
        let c = &self.config;
        let data: Vec<f32> = windows
            .iter()
            .flat_map(|w| w.rows.iter().copied())
            .collect();
        let input =
            ort::value::Tensor::from_array(([windows.len(), c.rows(), c.max_length], data))?;
        let outputs = self.session.run(ort::inputs!["rows"=>input])?;
        let (shape, values) = outputs["logits"].try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            shape.as_ref() == [windows.len() as i64, c.max_length as i64, 5],
            "unexpected ONNX output shape"
        );
        Ok(values.to_vec())
    }
}
