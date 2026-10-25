use clap::{CommandFactory, Parser};
use deepconsensus_rs::{
    cli::{Cli, Command},
    config::ModelConfig,
    dataset::DatasetWriter,
    inference, onnx,
    preprocess::Window,
};
use std::{path::Path, process::Output};

fn run(binary: &str, args: &[&str]) -> Output {
    std::process::Command::new(binary)
        .args(args)
        .env("RAYON_NUM_THREADS", "2")
        .output()
        .unwrap()
}

fn success(output: Output) -> Output {
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn tiny_config() -> ModelConfig {
    ModelConfig {
        max_length: 4,
        max_passes: 1,
        hidden_size: 8,
        num_heads: 2,
        num_hidden_layers: 1,
        filter_size: 16,
        attention_dropout: 0.0,
        ..Default::default()
    }
}

fn input_bams(root: &Path) {
    use rust_htslib::bam::{
        self,
        header::HeaderRecord,
        record::{Aux, Cigar, CigarString},
    };
    let mut header = bam::Header::new();
    header.push_record(
        HeaderRecord::new(b"SQ")
            .push_tag(b"SN", "movie/1/ccs")
            .push_tag(b"LN", 4),
    );
    let mut subreads =
        bam::Writer::from_path(root.join("subreads.bam"), &header, bam::Format::Bam).unwrap();
    let mut r = bam::Record::new();
    r.set(
        b"movie/1/0_4",
        Some(&CigarString(vec![Cigar::Match(4)])),
        b"ACGT",
        &[30; 4],
    );
    r.set_tid(0);
    r.set_pos(0);
    r.set_flags(0);
    r.push_aux(b"pw", Aux::ArrayU8((&[1u8; 4][..]).into()))
        .unwrap();
    r.push_aux(b"ip", Aux::ArrayU8((&[1u8; 4][..]).into()))
        .unwrap();
    r.push_aux(b"sn", Aux::ArrayFloat((&[2f32; 4][..]).into()))
        .unwrap();
    subreads.write(&r).unwrap();
    let mut ccs = bam::Writer::from_path(root.join("ccs.bam"), &header, bam::Format::Bam).unwrap();
    let mut r = bam::Record::new();
    r.set(b"movie/1/ccs", None, b"ACGT", &[30; 4]);
    r.set_tid(-1);
    r.set_pos(-1);
    r.set_flags(4);
    ccs.write(&r).unwrap();
}

#[test]
fn command_surface_and_defaults() {
    let mut cli = Cli::command();
    cli.debug_assert();
    cli = Cli::command();
    let commands: Vec<_> = cli.get_subcommands().map(|c| c.get_name()).collect();
    assert_eq!(
        commands,
        ["preprocess", "inference", "train", "evaluate", "benchmark"]
    );
    for name in ["init", "import-tf", "inspect-tf", "export-onnx"] {
        assert!(Cli::try_parse_from(["deepconsensus-rs", name]).is_err());
    }
    let c = Cli::try_parse_from([
        "dc",
        "inference",
        "--subreads",
        "sub.bam",
        "--ccs",
        "ccs.bam",
        "--model",
        "native",
        "--output",
        "out.fastq",
    ])
    .unwrap();
    assert_eq!(c.cpu_threads, 0);
    match c.command {
        Command::Inference {
            device,
            batch_size,
            block_molecules,
            skip_above,
            min_quality,
            min_length,
            max_quality,
            ccs_calibration,
            ..
        } => {
            assert_eq!(device, "cpu");
            assert_eq!((batch_size, block_molecules), (32, 16));
            assert_eq!(
                (skip_above, min_quality, min_length, max_quality),
                (45.0, 0.0, 1, 93)
            );
            assert_eq!(ccs_calibration, "skip");
        }
        _ => panic!("expected inference"),
    }
    assert!(
        Cli::try_parse_from([
            "dc",
            "run",
            "--subreads",
            "a",
            "--ccs",
            "b",
            "--model",
            "m",
            "--output",
            "o"
        ])
        .is_ok()
    );
}

#[test]
fn invalid_values_and_truth_are_rejected() {
    let base = [
        "dc",
        "inference",
        "--subreads",
        "a",
        "--ccs",
        "b",
        "--model",
        "m",
        "--output",
        "o",
    ];
    for (key, value) in [
        ("--batch-size", "0"),
        ("--block-molecules", "0"),
        ("--limit", "0"),
        ("--device", "cuda:-1"),
        ("--device", "cuda"),
        ("--skip-above", "NaN"),
        ("--min-quality", "inf"),
        ("--max-quality", "94"),
        ("--truth", "truth.bam"),
    ] {
        let args: Vec<_> = base.iter().copied().chain([key, value]).collect();
        assert!(Cli::try_parse_from(args).is_err(), "accepted {key}={value}");
    }
}

#[test]
fn platform_specific_provider_options() {
    use clap::ValueEnum;
    let names: Vec<_> = deepconsensus_rs::cli::Provider::value_variants()
        .iter()
        .map(|p| p.name())
        .collect();
    assert!(names.contains(&"cpu"));
    assert_eq!(
        names.contains(&"coreml"),
        cfg!(all(target_os = "macos", feature = "onnx-coreml"))
    );
    assert_eq!(
        names.contains(&"cuda"),
        cfg!(all(target_os = "linux", feature = "onnx-cuda"))
    );
    assert_eq!(
        names.contains(&"tensorrt"),
        cfg!(all(target_os = "linux", feature = "tensorrt"))
    );
    const {
        assert!(
            !cfg!(feature = "metal") || (cfg!(feature = "onnx") && cfg!(feature = "onnx-coreml"))
        );
        assert!(!cfg!(feature = "cuda") || (cfg!(feature = "onnx") && cfg!(feature = "onnx-cuda")));
    }
    let main = env!("CARGO_BIN_EXE_deepconsensus-rs");
    let helper = env!("CARGO_BIN_EXE_deepconsensus-model");
    let help = success(run(main, &["inference", "--help"]));
    let text = String::from_utf8(help.stdout).unwrap();
    assert!(
        text.contains("--provider") && text.contains("--config") && text.contains("--ort-threads")
    );
    assert!(text.contains(".onnx"));
    assert!(
        success(run(helper, &["--help"]))
            .stdout
            .windows(11)
            .any(|s| s == b"export-onnx")
    );
    assert!(
        Cli::try_parse_from([
            "dc",
            "train",
            "--data",
            "data.tfrecord",
            "--output",
            "out",
            "--export-onnx"
        ])
        .is_ok()
    );
    let base = [
        "dc",
        "inference",
        "--subreads",
        "a",
        "--ccs",
        "b",
        "--model",
        "m",
        "--output",
        "o",
        "--provider",
    ];
    for p in ["cpu", "cuda", "tensorrt", "coreml"] {
        assert_eq!(
            Cli::try_parse_from(base.into_iter().chain([p])).is_ok(),
            names.contains(&p),
            "provider {p}"
        );
    }
}

#[test]
fn optional_export_path_parsing() {
    for (tail, expected) in [
        (vec![], None),
        (vec!["--export-onnx"], Some(None)),
        (
            vec!["--export-onnx", "exports/trained.onnx"],
            Some(Some("exports/trained.onnx".into())),
        ),
    ] {
        let args: Vec<_> = ["dc", "train", "--data", "data.tfrecord", "--output", "out"]
            .into_iter()
            .chain(tail)
            .collect();
        match Cli::try_parse_from(args).unwrap().command {
            Command::Train { export_onnx, .. } => assert_eq!(export_onnx, expected),
            _ => panic!("expected train"),
        }
    }
}

#[test]
fn onnx_provider_selection_and_conflicts() {
    assert_eq!(inference::onnx_provider("cpu", None).unwrap(), "cpu");
    assert_eq!(
        inference::onnx_provider("metal", None).unwrap(),
        "coreml:gpu"
    );
    assert_eq!(
        inference::onnx_provider("metal", Some("coreml")).unwrap(),
        "coreml:gpu"
    );
    assert_eq!(
        inference::onnx_provider("cpu", Some("coreml")).unwrap(),
        "coreml:cpu"
    );
    assert_eq!(inference::onnx_provider("cuda:2", None).unwrap(), "cuda:2");
    assert_eq!(
        inference::onnx_provider("cuda:3", Some("tensorrt")).unwrap(),
        "tensorrt:3"
    );
    for (device, provider) in [
        ("metal", Some("cpu")),
        ("metal", Some("cuda")),
        ("cuda:1", Some("coreml")),
        ("cpu", Some("cuda")),
        ("cpu", Some("tensorrt")),
        ("cuda:1", Some("cpu")),
        ("cuda:-1", None),
    ] {
        assert!(inference::onnx_provider(device, provider).is_err());
    }
    assert_eq!(
        onnx::config_path(Path::new("a/model.onnx")),
        Path::new("a/model.params.json")
    );
    assert!(onnx::validate_export_path(Path::new("params.json")).is_err());
}

#[test]
fn cli_training_inference_and_model_helper() {
    let root = tempfile::tempdir().unwrap();
    let c = tiny_config();
    let params = root.path().join("params.json");
    c.save(&params).unwrap();
    let data = root.path().join("train.tfrecord.gz");
    let mut writer = DatasetWriter::new(&data, &c).unwrap();
    for i in 0..2 {
        writer
            .push(&Window {
                name: format!("movie/{i}/ccs"),
                window_pos: 0,
                length: 4,
                passes: 1,
                rows: (0..c.rows() * 4).map(|i| (i % 5) as f32).collect(),
                label: Some(vec![1, 0, 2, 3]),
                ccs: vec![1, 0, 2, 3],
                qualities: vec![30; 4],
                overflow: false,
            })
            .unwrap();
    }
    writer.finish().unwrap();
    let directory = root.path().join("trained");
    let initial = root.path().join("initial");
    let main = env!("CARGO_BIN_EXE_deepconsensus-rs");
    let helper = env!("CARGO_BIN_EXE_deepconsensus-model");
    success(run(
        helper,
        &[
            "init",
            "--config",
            params.to_str().unwrap(),
            "--output",
            initial.to_str().unwrap(),
        ],
    ));
    assert!(initial.join("model.burnpack").is_file());
    let mut train = vec![
        "--cpu-threads",
        "2",
        "train",
        "--data",
        data.to_str().unwrap(),
        "--config",
        params.to_str().unwrap(),
        "--output",
        directory.to_str().unwrap(),
        "--batch-size",
        "2",
    ];
    train.push("--export-onnx");
    success(run(main, &train));
    let protected = run(
        helper,
        &[
            "init",
            "--config",
            params.to_str().unwrap(),
            "--output",
            directory.to_str().unwrap(),
        ],
    );
    assert!(!protected.status.success());
    assert!(String::from_utf8_lossy(&protected.stderr).contains("optimizer/training state"));
    let graph = directory.join("model.onnx");
    {
        assert!(graph.is_file());
        assert_eq!(
            serde_json::to_value(ModelConfig::load(onnx::config_path(&graph)).unwrap()).unwrap(),
            serde_json::to_value(&c).unwrap()
        );
        let second = root.path().join("exports/standalone.onnx");
        success(run(
            helper,
            &[
                "export-onnx",
                "--model",
                directory.to_str().unwrap(),
                "--output",
                second.to_str().unwrap(),
            ],
        ));
        assert!(onnx::config_path(&second).is_file());
        let custom = root.path().join("exports/resumed.onnx");
        success(run(
            main,
            &[
                "--cpu-threads",
                "2",
                "train",
                "--data",
                data.to_str().unwrap(),
                "--resume",
                directory.to_str().unwrap(),
                "--output",
                directory.to_str().unwrap(),
                "--epochs",
                "2",
                "--batch-size",
                "2",
                "--export-onnx",
                custom.to_str().unwrap(),
            ],
        ));
        assert!(custom.is_file() && onnx::config_path(&custom).is_file());
        success(run(
            helper,
            &["export-onnx", "--model", directory.to_str().unwrap()],
        ));
    }
    input_bams(root.path());
    let subreads = root.path().join("subreads.bam");
    let ccs = root.path().join("ccs.bam");
    let native_output = root.path().join("native.fastq");
    let args = [
        "--cpu-threads",
        "2",
        "inference",
        "--subreads",
        subreads.to_str().unwrap(),
        "--ccs",
        ccs.to_str().unwrap(),
        "--model",
        directory.to_str().unwrap(),
        "--output",
        native_output.to_str().unwrap(),
        "--skip-above",
        "0",
        "--threads",
        "1",
    ];
    let result = success(run(main, &args));
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["molecules"], 1);
    assert_eq!(report["written"], 1);
    assert_eq!(report["model_windows"], 1);
    assert!(native_output.is_file());
    assert!(!std::fs::read(&native_output).unwrap().is_empty());
    #[cfg(feature = "onnx")]
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        let output = root.path().join("onnx.fastq");
        let mut args = args.to_vec();
        let index = args.iter().position(|&a| a == "--model").unwrap() + 1;
        args[index] = graph.to_str().unwrap();
        let index = args.iter().position(|&a| a == "--output").unwrap() + 1;
        args[index] = output.to_str().unwrap();
        success(run(main, &args));
        assert_eq!(
            std::fs::read(&native_output).unwrap(),
            std::fs::read(&output).unwrap()
        );
        std::fs::remove_file(onnx::config_path(&graph)).unwrap();
        let error = run(main, &args);
        assert!(!error.status.success());
        assert!(String::from_utf8_lossy(&error.stderr).contains("--config"));
        let params = directory.join("params.json");
        args.extend(["--config", params.to_str().unwrap()]);
        success(run(main, &args));
        #[cfg(all(target_os = "macos", feature = "onnx-coreml"))]
        if std::env::var("DC_COREML_CPU_TEST").as_deref() == Ok("1") {
            args.extend(["--provider", "coreml"]);
            let result = success(run(main, &args));
            assert!(String::from_utf8_lossy(&result.stderr).contains("CoreML selected (CPUOnly)"));
            assert_eq!(
                std::fs::read(&native_output).unwrap(),
                std::fs::read(&output).unwrap()
            );
        }
    }
}
