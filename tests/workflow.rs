use deepconsensus_rs::{
    checkpoint,
    config::ModelConfig,
    dataset::{DatasetReader, DatasetWriter},
    model,
    preprocess::Window,
    training::{self, TrainOptions},
};

#[test]
fn training_resume_export_and_evaluation() {
    let root = tempfile::tempdir().unwrap();
    let c = ModelConfig {
        max_length: 4,
        max_passes: 1,
        hidden_size: 8,
        num_heads: 2,
        num_hidden_layers: 1,
        filter_size: 16,
        attention_dropout: 0.0,
        band_width: Some(2),
        ..Default::default()
    };
    let data = root.path().join("tiny.tfrecord.gz");
    let mut writer = DatasetWriter::new(&data, &c).unwrap();
    for i in 0..4 {
        let mut rows = vec![0f32; c.rows() * 4];
        rows[..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        rows[16..20].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        writer
            .push(&Window {
                name: format!("m/{i}/ccs"),
                window_pos: 0,
                length: 4,
                passes: 1,
                rows,
                label: Some(vec![1, 2, 3, 4]),
                ccs: vec![1, 2, 3, 4],
                qualities: vec![30; 4],
                overflow: false,
            })
            .unwrap();
    }
    writer.finish().unwrap();
    let directory = root.path().join("model");
    let mut options = TrainOptions {
        datasets: vec![data.clone()],
        output: directory.clone(),
        resume: None,
        teacher: None,
        epochs: 1,
        batch_size: 2,
        learning_rate: 0.003,
        end_learning_rate: 0.001,
        warmup_steps: 0,
        weight_decay: 0.0,
        seed: 1,
        shuffle_buffer: 4,
        max_examples: None,
        device: "cpu".into(),
        distillation_weight: 1.0,
    };
    training::train(&c, &options).unwrap();
    let d = model::device("cpu").unwrap();
    let first = checkpoint::load(&directory, &d).unwrap();
    let mut r = DatasetReader::new(&data, &c).unwrap();
    let w = r.next_window().unwrap().unwrap();
    let before = first
        .forward(
            model::batch(std::slice::from_ref(&w), &c, &d).unwrap(),
            false,
        )
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    options.resume = Some(directory.clone());
    options.epochs = 2;
    training::train(&c, &options).unwrap();
    let second = checkpoint::load(&directory, &d).unwrap();
    let after = second
        .forward(model::batch(&[w], &c, &d).unwrap(), false)
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    assert_ne!(before, after);
    let metrics = training::evaluate(&second, &[data], 2, &d).unwrap();
    assert_eq!(metrics["examples"], 4);
    assert!(metrics["mean_alignment_loss"].as_f64().unwrap().is_finite());
    deepconsensus_rs::onnx::export(&second, &root.path().join("model.onnx")).unwrap();
    #[cfg(feature = "onnx")]
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        for optimized in [false, true] {
            let path = root.path().join(format!("model-{optimized}.onnx"));
            deepconsensus_rs::onnx::export_with_options(&second, &path, optimized).unwrap();
            let mut ort = deepconsensus_rs::onnx::OrtEngine::new(&path, &c, "cpu", 1).unwrap();
            let mut r = DatasetReader::new(options.datasets[0].as_path(), &c).unwrap();
            let w = r.next_window().unwrap().unwrap();
            for size in [1, 3] {
                let windows = vec![w.clone(); size];
                let expected = second
                    .forward(model::batch(&windows, &c, &d).unwrap(), false)
                    .to_data()
                    .try_to_vec::<f32>()
                    .unwrap();
                let actual = ort.predict(&windows).unwrap();
                let error = expected
                    .iter()
                    .zip(actual)
                    .map(|(&a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    error < 1e-4,
                    "ORT error {error}, optimized={optimized}, batch={size}"
                );
            }
        }
    }
}

#[test]
fn real_bam_fixture_if_available() {
    let Some(root) = std::env::var_os("DC_FIXTURE_ROOT") else {
        return;
    };
    let root = std::path::PathBuf::from(root);
    let c = ModelConfig::default();
    let mut reader = deepconsensus_rs::bam::MoleculeReader::open(
        root.join("subreads_to_ccs.bam"),
        root.join("ccs.bam"),
        None,
        1,
        5,
    )
    .unwrap();
    let mut count = 0;
    while let Some(m) = reader.next_molecule().unwrap() {
        let w = deepconsensus_rs::preprocess::windows(&m, &c, false).unwrap();
        assert!(!w.is_empty());
        for x in w {
            assert_eq!(x.rows.len(), 8600);
            assert_eq!(x.qualities.len(), 100);
        }
        count += 1;
    }
    assert_eq!(count, 10);
}
