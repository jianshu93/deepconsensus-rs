#![cfg(all(target_os = "macos", feature = "metal"))]

use burn::{
    module::Param,
    tensor::{Device, Tensor, TensorData},
};
use deepconsensus_rs::{
    alignment, checkpoint,
    config::ModelConfig,
    dataset::{DatasetReader, DatasetWriter},
    model,
    preprocess::Window,
    training::{self, TrainOptions},
};

#[test]
#[ignore = "requires a macOS Metal GPU"]
fn metal_alignment_values_and_gradients() {
    let device = Device::autodiff(model::device("metal").unwrap());
    eprintln!("Metal device: {device:?}");
    let values: Vec<f32> = (0..40).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.2).collect();
    let labels = vec![vec![1, 0, 2, 3], vec![2, 4]];
    for band in [None, Some(2)] {
        let c = ModelConfig {
            max_length: 4,
            band_width: band,
            loss_reg: Some(0.3),
            del_cost: 2.0,
            ..Default::default()
        };
        let x = Tensor::<3>::from_data(TensorData::new(values.clone(), [2, 4, 5]), &device)
            .require_grad();
        let loss = alignment::wavefront_loss(x.clone(), &labels, &c).unwrap();
        let value = loss.clone().into_scalar::<f32>() as f64;
        let grads = loss.backward();
        let gradient = x
            .grad(&grads)
            .unwrap()
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        let a = alignment::alignment_loss(&values[..20], &labels[0], &c).unwrap();
        let b = alignment::alignment_loss(&values[20..], &labels[1], &c).unwrap();
        assert!(
            (value - (a.loss + b.loss) / 2.0).abs() < 2e-5,
            "Metal loss {value}"
        );
        for (&g, expected) in gradient.iter().zip(a.gradient.iter().chain(&b.gradient)) {
            assert!(
                (g - expected / 2.0).abs() < 2e-5,
                "Metal gradient {g} != {}",
                expected / 2.0
            );
        }
    }
}

#[test]
#[ignore = "requires a macOS Metal GPU"]
fn metal_forward_training_resume_and_checkpoint() {
    let root = tempfile::tempdir().unwrap();
    let c = ModelConfig {
        max_length: 4,
        max_passes: 1,
        hidden_size: 8,
        num_heads: 2,
        num_hidden_layers: 1,
        filter_size: 16,
        attention_dropout: 0.0,
        ..Default::default()
    };
    let cpu = model::device("cpu").unwrap();
    cpu.seed(1);
    let mut original = model::DeepConsensus::new(&c, &cpu).unwrap();
    original.blocks[0].attention_alpha = Param::from_tensor(Tensor::full([1], 0.3, &cpu));
    original.blocks[0].ffn_alpha = Param::from_tensor(Tensor::full([1], 0.4, &cpu));
    let initial = root.path().join("initial");
    checkpoint::save(&original, &initial).unwrap();
    let window = Window {
        name: "metal/1/ccs".into(),
        window_pos: 0,
        length: 4,
        passes: 1,
        rows: (0..c.rows() * 4).map(|i| (i % 5) as f32).collect(),
        label: Some(vec![1, 0, 2, 3]),
        ccs: vec![1, 0, 2, 3],
        qualities: vec![30; 4],
        overflow: false,
    };
    let reference = original
        .forward(
            model::batch(std::slice::from_ref(&window), &c, &cpu).unwrap(),
            false,
        )
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    let gpu = model::device("metal").unwrap();
    let loaded = checkpoint::load(&initial, &gpu).unwrap();
    let actual = loaded
        .forward(
            model::batch(std::slice::from_ref(&window), &c, &gpu).unwrap(),
            false,
        )
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    let max = reference
        .iter()
        .zip(&actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max < 2e-4, "Metal forward error {max}");
    let data = root.path().join("train.tfrecord.gz");
    let mut writer = DatasetWriter::new(&data, &c).unwrap();
    for i in 0..2 {
        let mut w = window.clone();
        w.name = format!("metal/{i}/ccs");
        writer.push(&w).unwrap();
    }
    writer.finish().unwrap();
    let output = root.path().join("trained");
    let mut options = TrainOptions {
        datasets: vec![data.clone()],
        output: output.clone(),
        resume: Some(initial),
        teacher: None,
        epochs: 1,
        batch_size: 2,
        learning_rate: 0.003,
        end_learning_rate: 0.003,
        warmup_steps: 0,
        weight_decay: 0.0,
        seed: 1,
        shuffle_buffer: 1,
        max_examples: None,
        device: "metal".into(),
        distillation_weight: 0.0,
    };
    training::train(&c, &options).unwrap();
    assert!(output.join("optimizer.burnpack").exists());
    let trained = checkpoint::load(&output, &cpu).unwrap();
    let after = trained
        .forward(
            model::batch(std::slice::from_ref(&window), &c, &cpu).unwrap(),
            false,
        )
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    assert_ne!(reference, after, "Metal optimizer did not update the model");
    assert!(after.iter().all(|x| x.is_finite()));
    options.resume = Some(output.clone());
    options.epochs = 2;
    training::train(&c, &options).unwrap();
    let state: training::TrainState =
        serde_json::from_reader(std::fs::File::open(output.join("training-state.json")).unwrap())
            .unwrap();
    assert_eq!((state.epoch, state.step), (2, 2));
    let trained = checkpoint::load(&output, &gpu).unwrap();
    let report = training::evaluate(&trained, &[data], 2, &gpu).unwrap();
    assert_eq!(report["examples"], 2);
    assert!(report["mean_alignment_loss"].as_f64().unwrap().is_finite());
}

#[test]
#[ignore = "requires a macOS Metal GPU and DC_METAL_MODEL / DC_METAL_DATA"]
fn published_model_metal_matches_cpu() {
    let model_path = std::env::var("DC_METAL_MODEL")
        .expect("set DC_METAL_MODEL to the published native model directory");
    let data =
        std::env::var("DC_METAL_DATA").expect("set DC_METAL_DATA to a real labeled TFRecord");
    let cpu = model::device("cpu").unwrap();
    let gpu = model::device("metal").unwrap();
    let original = checkpoint::load(model_path.as_ref(), &cpu).unwrap();
    let loaded = checkpoint::load(model_path.as_ref(), &gpu).unwrap();
    let c = &original.config;
    let mut reader = DatasetReader::new(data.as_ref(), c).unwrap();
    let windows = (0..3)
        .map(|_| {
            reader
                .next_window()
                .unwrap()
                .expect("need three real windows")
        })
        .collect::<Vec<_>>();
    let a = original
        .forward(model::batch(&windows, c, &cpu).unwrap(), false)
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    let b = loaded
        .forward(model::batch(&windows, c, &gpu).unwrap(), false)
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    let max = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let argmax = |x: &[f32]| {
        x.iter()
            .enumerate()
            .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
            .unwrap()
            .0
    };
    let changed = a
        .as_chunks::<5>()
        .0
        .iter()
        .zip(b.as_chunks::<5>().0.iter())
        .filter(|(a, b)| argmax(*a) != argmax(*b))
        .count();
    eprintln!("Published model Metal max_logit_difference={max} changed_calls={changed}");
    assert!(max < 3e-4, "Metal published model logit error {max}");
    assert_eq!(changed, 0);
}
