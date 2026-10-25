#![cfg(all(target_os = "macos", feature = "onnx-coreml"))]

use burn::{module::Param, tensor::Tensor};
use deepconsensus_rs::{config::ModelConfig, model, onnx, preprocess::Window};

fn check(provider: &str) {
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
    let device = model::device("cpu").unwrap();
    device.seed(11);
    let mut m = model::DeepConsensus::new(&c, &device).unwrap();
    m.blocks[0].attention_alpha = Param::from_tensor(Tensor::full([1], 0.3, &device));
    m.blocks[0].ffn_alpha = Param::from_tensor(Tensor::full([1], 0.4, &device));
    let path = root.path().join("model.onnx");
    onnx::export(&m, &path).unwrap();
    let mut reference = onnx::OrtEngine::new(&path, &c, "cpu", 1).unwrap();
    let mut actual = onnx::OrtEngine::new(&path, &c, provider, 1).unwrap();
    let w = Window {
        name: "coreml/1/ccs".into(),
        window_pos: 0,
        length: 4,
        passes: 1,
        rows: (0..c.rows() * 4).map(|i| (i % 5) as f32).collect(),
        label: None,
        ccs: vec![1, 0, 2, 3],
        qualities: vec![30; 4],
        overflow: false,
    };
    for n in [1, 3] {
        let windows = vec![w.clone(); n];
        let expected = reference.predict(&windows).unwrap();
        let values = actual.predict(&windows).unwrap();
        assert_eq!(expected.len(), values.len());
        assert!(values.iter().all(|v| v.is_finite()));
        let error = expected
            .iter()
            .zip(values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("{provider} batch={n} max_logit_difference={error}");
        assert!(error < 1e-3, "CoreML logit error {error}");
        let argmax = |x: &[f32]| {
            x.iter()
                .enumerate()
                .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
                .unwrap()
                .0
        };
        for (a, b) in expected
            .as_chunks::<5>()
            .0
            .iter()
            .zip(values.as_chunks::<5>().0.iter())
        {
            assert_eq!(argmax(a), argmax(b), "CoreML changed a base call");
        }
    }
}

#[test]
#[ignore = "requires a macOS ORT runtime with CoreML and model compilation permission"]
fn coreml_cpu_matches_ort_cpu() {
    check("coreml:cpu");
}

#[test]
#[ignore = "requires an accessible Apple GPU and CoreML-enabled macOS ORT runtime"]
fn coreml_gpu_matches_ort_cpu() {
    check("coreml:gpu");
}
