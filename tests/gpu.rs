#![cfg(all(target_os = "linux", feature = "cuda"))]
use burn::{
    module::{Module, Param},
    tensor::{Device, Tensor, TensorData},
};
use deepconsensus_rs::{alignment, checkpoint, config::ModelConfig, model, preprocess::Window};

#[test]
#[ignore = "requires a Linux NVIDIA GPU"]
fn cuda_forward_and_wavefront_backward() {
    let c = ModelConfig {
        max_length: 8,
        max_passes: 2,
        hidden_size: 16,
        num_heads: 2,
        num_hidden_layers: 1,
        filter_size: 32,
        attention_dropout: 0.0,
        band_width: Some(2),
        ..Default::default()
    };
    let cpu = model::device("cpu").unwrap();
    cpu.seed(1);
    let root = tempfile::tempdir().unwrap();
    let mut original = model::DeepConsensus::new(&c, &cpu).unwrap();
    original.blocks[0].attention_alpha = Param::from_tensor(Tensor::full([1], 0.3, &cpu));
    original.blocks[0].ffn_alpha = Param::from_tensor(Tensor::full([1], 0.4, &cpu));
    checkpoint::save(&original, root.path()).unwrap();
    let gpu = Device::autodiff(model::device("cuda:0").unwrap());
    let loaded = checkpoint::load(root.path(), &gpu).unwrap();
    let w = Window {
        name: "gpu/1/ccs".into(),
        window_pos: 0,
        length: 8,
        passes: 2,
        rows: vec![1.0; c.rows() * 8],
        label: Some(vec![1, 2, 3, 4, 1, 2, 3, 4]),
        ccs: vec![1; 8],
        qualities: vec![30; 8],
        overflow: false,
    };
    let reference = original
        .forward(
            model::batch(std::slice::from_ref(&w), &c, &cpu).unwrap(),
            false,
        )
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    let logits = loaded.forward(
        model::batch(std::slice::from_ref(&w), &c, &gpu).unwrap(),
        false,
    );
    let actual = logits.to_data().try_to_vec::<f32>().unwrap();
    let max = reference
        .iter()
        .zip(&actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max < 2e-4, "CUDA logit error {max}");
    let inference = loaded.valid();
    let options = model::ExecutionOptions {
        folded_embeddings: true,
        fused_qkv: true,
        fused_attention: true,
        flattened_linears: true,
    };
    let fused = inference
        .forward_with_options(
            model::batch(
                std::slice::from_ref(&w),
                &c,
                &model::device("cuda:0").unwrap(),
            )
            .unwrap(),
            false,
            options,
        )
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    let max = actual
        .iter()
        .zip(fused)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max < 2e-4, "CUDA fused logit error {max}");
    let expected = alignment::alignment_loss(&actual, w.label.as_ref().unwrap(), &c).unwrap();
    let loss = alignment::wavefront_loss(logits, &[w.label.unwrap()], &c).unwrap();
    assert!((loss.clone().into_scalar::<f32>() as f64 - expected.loss).abs() < 1e-4);
    let gradients = loss.backward();
    let grad = loaded
        .classifier
        .weight
        .val()
        .grad(&gradients)
        .unwrap()
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    assert!(grad.iter().all(|x| x.is_finite()));
    assert!(grad.iter().any(|x| *x != 0.0));
    // Direct logit gradients also check the device DP independently of the model.
    let x = Tensor::<3>::from_data(TensorData::new(actual, [1, 8, 5]), &gpu).require_grad();
    let loss = alignment::wavefront_loss(x.clone(), &[vec![1, 2, 3, 4, 1, 2, 3, 4]], &c).unwrap();
    let grads = loss.backward();
    let got = x
        .grad(&grads)
        .unwrap()
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    for (a, b) in got.iter().zip(expected.gradient) {
        assert!((a - b).abs() < 1e-4);
    }
}
