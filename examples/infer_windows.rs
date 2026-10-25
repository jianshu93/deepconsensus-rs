use anyhow::{Result, ensure};
use clap::Parser;
use deepconsensus_rs::{checkpoint, dataset::DatasetReader, model, preprocess::VOCAB};
use std::{
    collections::HashSet,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value_t = 64)]
    batch_size: usize,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    #[arg(long)]
    max_windows: Option<usize>,
}

fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(a.batch_size > 0 && a.repeats > 0, "invalid dimensions");
    rayon::ThreadPoolBuilder::new()
        .num_threads(a.threads)
        .build_global()?;
    let device = model::device(&a.device)?;
    let m = checkpoint::load(&a.model, &device)?;
    let c = &m.config;
    let mut reader = DatasetReader::new(&a.data, c)?;
    let mut windows = Vec::new();
    let mut unique = HashSet::new();
    while a.max_windows.is_none_or(|n| windows.len() < n) {
        let Some(w) = reader.next_window()? else {
            break;
        };
        ensure!(
            unique.insert((w.name.clone(), w.window_pos)),
            "duplicate window"
        );
        windows.push(w);
    }
    ensure!(!windows.is_empty(), "empty input");
    let _ = m
        .forward(
            model::batch(&windows[..a.batch_size.min(windows.len())], c, &device)?,
            false,
        )
        .to_data();
    let mut samples = Vec::new();
    let mut logits = Vec::new();
    for _ in 0..a.repeats {
        logits.clear();
        let timer = Instant::now();
        for chunk in windows.chunks(a.batch_size) {
            logits.extend(
                m.forward(model::batch(chunk, c, &device)?, false)
                    .to_data()
                    .try_to_vec::<f32>()?,
            );
        }
        let seconds = timer.elapsed().as_secs_f64();
        samples.push(seconds);
        eprintln!("windows={} forward_seconds={seconds:.6}", windows.len());
    }
    ensure!(
        logits.iter().all(|x| x.is_finite()),
        "non-finite model output"
    );
    if let Some(parent) = a.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut raw = BufWriter::new(std::fs::File::create(
        a.output.with_extension("logits.f32"),
    )?);
    for x in &logits {
        raw.write_all(&x.to_le_bytes())?;
    }
    raw.flush()?;
    let mut fasta = BufWriter::new(std::fs::File::create(a.output.with_extension("fasta"))?);
    for (w, x) in windows.iter().zip(logits.chunks_exact(c.max_length * 5)) {
        writeln!(fasta, ">{}:{}", w.name, w.window_pos)?;
        for p in x.as_chunks::<5>().0 {
            let id = p
                .iter()
                .enumerate()
                .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
                .unwrap()
                .0;
            if id != 0 {
                fasta.write_all(&[VOCAB[id]])?;
            }
        }
        fasta.write_all(b"\n")?;
    }
    fasta.flush()?;
    let report = serde_json::json!({"device":a.device,"windows":windows.len(),"unique_windows":unique.len(),
        "batch_size":a.batch_size,"samples_seconds":samples,
        "timed_scope":"Batch construction, model, synchronized output transfers; load/read/warmup/write excluded",
        "output_note":"Sampled window FASTA, not stitched molecule consensus"});
    std::fs::write(
        a.output.with_extension("json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
