use crate::{
    alignment::alignment_loss,
    checkpoint,
    config::ModelConfig,
    dataset::DatasetReader,
    model::{self, DeepConsensus},
    preprocess::Window,
};
use anyhow::{Result, ensure};
use burn::{
    optim::{AdamWConfig, GradientsParams},
    prelude::*,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainOptions {
    pub datasets: Vec<PathBuf>,
    pub output: PathBuf,
    pub resume: Option<PathBuf>,
    pub teacher: Option<PathBuf>,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub end_learning_rate: f64,
    pub warmup_steps: usize,
    pub weight_decay: f32,
    pub seed: u64,
    pub shuffle_buffer: usize,
    pub max_examples: Option<usize>,
    pub device: String,
    pub distillation_weight: f64,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TrainState {
    pub epoch: usize,
    pub step: usize,
}

pub fn tensor_loss(
    logits: Tensor<3>,
    labels: &[Vec<u8>],
    c: &ModelConfig,
) -> Result<(Tensor<1>, f64, usize)> {
    let [b, n, k] = logits.dims();
    ensure!(labels.len() == b && k == 5, "loss batch mismatch");
    let values = logits.to_data().try_to_vec::<f32>()?;
    let result: Vec<_> = values
        .par_chunks_exact(n * k)
        .zip(labels.par_iter())
        .map(|(x, y)| alignment_loss(x, y, c))
        .collect::<Result<_>>()?;
    let loss = result.iter().map(|r| r.loss).sum::<f64>() / b as f64;
    let cells = result.iter().map(|r| r.active_cells).sum();
    let gradients: Vec<f32> = result
        .into_iter()
        .flat_map(|r| r.gradient.into_iter().map(|g| g / b as f32))
        .collect();
    let grad = Tensor::<3>::from_data(TensorData::new(gradients, [b, n, k]), &logits.device());
    // Exact first-order VJP of the CPU soft-DP; no straight-through approximation.
    let linear = (logits * grad).sum();
    let stop = linear.clone().detach();
    Ok((linear - stop + loss, loss, cells))
}

struct Stream {
    paths: Vec<PathBuf>,
    position: usize,
    reader: Option<DatasetReader>,
    config: ModelConfig,
    loaded: usize,
    limit: Option<usize>,
}
impl Stream {
    fn new(paths: &[PathBuf], config: &ModelConfig, limit: Option<usize>) -> Self {
        Self {
            paths: paths.to_vec(),
            position: 0,
            reader: None,
            config: config.clone(),
            loaded: 0,
            limit,
        }
    }
    fn next(&mut self) -> Result<Option<Window>> {
        if self.limit.is_some_and(|n| self.loaded >= n) {
            return Ok(None);
        }
        loop {
            if let Some(r) = self.reader.as_mut()
                && let Some(w) = r.next_window()?
            {
                ensure!(
                    w.label.is_some(),
                    "training example {} has no label",
                    w.name
                );
                ensure!(!w.overflow, "overflow training window");
                self.loaded += 1;
                return Ok(Some(w));
            }
            if self.position == self.paths.len() {
                return Ok(None);
            }
            self.reader = Some(DatasetReader::new(
                &self.paths[self.position],
                &self.config,
            )?);
            self.position += 1;
        }
    }
}

pub fn train(c: &ModelConfig, o: &TrainOptions) -> Result<()> {
    ensure!(
        o.epochs > 0 && o.batch_size > 0 && o.shuffle_buffer > 0 && !o.datasets.is_empty(),
        "invalid training dimensions"
    );
    ensure!(
        o.learning_rate.is_finite()
            && o.learning_rate > 0.0
            && o.end_learning_rate.is_finite()
            && o.end_learning_rate >= 0.0,
        "invalid learning rate"
    );
    ensure!(
        o.distillation_weight.is_finite() && o.distillation_weight >= 0.0,
        "invalid distillation weight"
    );
    let base = model::device(&o.device)?;
    let device = Device::autodiff(base.clone());
    device.seed(o.seed);
    let mut state = TrainState::default();
    let mut model = if let Some(p) = &o.resume {
        let loaded = checkpoint::load(p, &device)?;
        ensure!(
            serde_json::to_value(&loaded.config)? == serde_json::to_value(c)?,
            "resume model configuration mismatch"
        );
        let path = p.join("training-state.json");
        if path.exists() {
            state = serde_json::from_reader(std::fs::File::open(path)?)?;
        }
        loaded
    } else {
        DeepConsensus::new(c, &device)?
    };
    let teacher = o
        .teacher
        .as_ref()
        .map(|p| checkpoint::load(p, &base))
        .transpose()?;
    if let Some(t) = &teacher {
        ensure!(
            t.config.rows() == c.rows() && t.config.max_length == c.max_length,
            "teacher input dimensions mismatch"
        );
    }
    let mut optimizer = AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.997)
        .with_epsilon(1e-9)
        .with_weight_decay(o.weight_decay)
        .init();
    if let Some(p) = &o.resume {
        let path = p.join("optimizer.burnpack");
        if path.exists() {
            optimizer = optimizer.load_on_device(path, &base)?;
        }
    }
    let mut count = Stream::new(&o.datasets, c, o.max_examples);
    let mut examples: usize = 0;
    while count.next()?.is_some() {
        examples += 1;
    }
    ensure!(examples > 0, "empty training dataset");
    let total_steps = o.epochs * examples.div_ceil(o.batch_size);
    ensure!(
        state.epoch < o.epochs,
        "checkpoint already reached requested epochs"
    );
    std::fs::create_dir_all(&o.output)?;
    std::fs::write(
        o.output.join("training-options.json"),
        serde_json::to_vec_pretty(o)?,
    )?;
    let timer = Instant::now();
    for epoch in state.epoch..o.epochs {
        let mut rng = StdRng::seed_from_u64(o.seed.wrapping_add(epoch as u64));
        let mut input = Stream::new(&o.datasets, c, o.max_examples);
        let mut buffer = Vec::new();
        for _ in 0..o.shuffle_buffer {
            if let Some(w) = input.next()? {
                buffer.push(w);
            } else {
                break;
            }
        }
        let mut epoch_loss = 0.0;
        let mut batches = 0;
        while !buffer.is_empty() {
            let mut windows = Vec::new();
            while windows.len() < o.batch_size && !buffer.is_empty() {
                let k = rng.gen_range(0..buffer.len());
                windows.push(buffer.swap_remove(k));
                if let Some(w) = input.next()? {
                    buffer.push(w);
                }
            }
            let labels: Vec<Vec<u8>> = windows.iter().map(|w| w.label.clone().unwrap()).collect();
            let rows = model::batch(&windows, c, &device)?;
            let logits = model.forward(rows, true);
            let (mut loss, value, cells) = if o.device.starts_with("cuda:") || o.device == "metal" {
                let loss = crate::alignment::wavefront_loss(logits.clone(), &labels, c)?;
                let value = loss.clone().into_scalar::<f32>() as f64;
                (loss, value, 0)
            } else {
                tensor_loss(logits.clone(), &labels, c)?
            };
            if let Some(t) = &teacher {
                let teacher_logits = t
                    .forward(model::batch(&windows, c, &base)?, false)
                    .to_data();
                let target = Tensor::<3>::from_data(teacher_logits, &device);
                loss = loss + (logits - target).powf_scalar(2.0).mean() * o.distillation_weight;
            }
            state.step += 1;
            let fraction = state.step.saturating_sub(o.warmup_steps) as f64
                / total_steps.saturating_sub(o.warmup_steps).max(1) as f64;
            let lr = if state.step < o.warmup_steps {
                o.learning_rate * state.step as f64 / o.warmup_steps as f64
            } else {
                o.end_learning_rate
                    + 0.5
                        * (o.learning_rate - o.end_learning_rate)
                        * (1.0 + (std::f64::consts::PI * fraction.min(1.0)).cos())
            };
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(lr, model, grads);
            epoch_loss += value;
            batches += 1;
            eprintln!(
                "epoch={} step={} alignment_loss={value:.6} lr={lr:.6e} active_cells={cells}",
                epoch + 1,
                state.step
            );
        }
        state.epoch = epoch + 1;
        checkpoint::save(&model, &o.output)?;
        optimizer.save(o.output.join("optimizer.burnpack"))?;
        std::fs::write(
            o.output.join("training-state.json"),
            serde_json::to_vec_pretty(&state)?,
        )?;
        eprintln!(
            "epoch={} mean_loss={:.6} elapsed_seconds={:.3}",
            state.epoch,
            epoch_loss / batches as f64,
            timer.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

pub fn evaluate(
    model: &DeepConsensus,
    paths: &[PathBuf],
    batch_size: usize,
    device: &Device,
) -> Result<serde_json::Value> {
    ensure!(batch_size > 0, "batch_size must be positive");
    let c = &model.config;
    let mut input = Stream::new(paths, c, None);
    let mut windows = Vec::new();
    let mut total = 0;
    let mut loss = 0.0;
    let mut edits = 0;
    let mut bases = 0;
    loop {
        while windows.len() < batch_size {
            if let Some(w) = input.next()? {
                windows.push(w);
            } else {
                break;
            }
        }
        if windows.is_empty() {
            break;
        }
        let values = model
            .forward(model::batch(&windows, c, device)?, false)
            .to_data()
            .try_to_vec::<f32>()?;
        for (x, w) in values.chunks_exact(c.max_length * 5).zip(&windows) {
            let truth: Vec<u8> = w
                .label
                .as_ref()
                .unwrap()
                .iter()
                .copied()
                .filter(|&x| x != 0)
                .collect();
            let pred: Vec<u8> = x
                .as_chunks::<5>()
                .0
                .iter()
                .map(|p| {
                    p.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .unwrap()
                        .0 as u8
                })
                .filter(|&x| x != 0)
                .collect();
            loss += alignment_loss(x, &truth, c)?.loss;
            edits += edit_distance(&truth, &pred);
            bases += truth.len();
            total += 1;
        }
        windows.clear();
    }
    ensure!(total > 0, "empty evaluation set");
    Ok(
        serde_json::json!({"examples":total,"mean_alignment_loss":loss/total as f64,"edit_errors":edits,"truth_bases":bases,"error_rate":edits as f64/bases.max(1) as f64}),
    )
}

pub fn edit_distance(a: &[u8], b: &[u8]) -> usize {
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, &x) in a.iter().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, &y) in b.iter().enumerate() {
            let old = row[j + 1];
            row[j + 1] = (diagonal + usize::from(x != y))
                .min(row[j] + 1)
                .min(old + 1);
            diagonal = old;
        }
    }
    row[b.len()]
}

pub fn prepare_dataset(
    input: &mut crate::bam::MoleculeReader,
    output: &Path,
    c: &ModelConfig,
    smart: bool,
    limit: Option<usize>,
) -> Result<usize> {
    let mut writer = crate::dataset::DatasetWriter::new(output, c)?;
    let mut molecules = 0;
    let mut examples = 0;
    while limit.is_none_or(|n| molecules < n) {
        let mut block = Vec::new();
        while block.len() < 32 && limit.is_none_or(|n| molecules + block.len() < n) {
            if let Some(m) = input.next_molecule()? {
                block.push(m);
            } else {
                break;
            }
        }
        if block.is_empty() {
            break;
        }
        let prepared: Vec<_> = block
            .par_iter()
            .map(|m| crate::preprocess::windows(m, c, smart))
            .collect::<Result<_>>()?;
        molecules += block.len();
        for windows in prepared {
            for w in windows {
                writer.push(&w)?;
                examples += 1;
            }
        }
    }
    writer.finish()?;
    Ok(examples)
}
