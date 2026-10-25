use anyhow::{Result, ensure};
use deepconsensus_rs::{checkpoint, dataset::DatasetReader, model};
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        args.len() == 4,
        "usage: dump_logits MODEL_DIR DATASET OUTPUT_JSON"
    );
    let d = model::device("cpu")?;
    let m = checkpoint::load(args[1].as_ref(), &d)?;
    let c = &m.config;
    let mut r = DatasetReader::new(args[2].as_ref(), c)?;
    let mut windows = Vec::new();
    for _ in 0..3 {
        if let Some(w) = r.next_window()? {
            windows.push(w);
        }
    }
    let logits = m
        .forward(model::batch(&windows, c, &d)?, false)
        .to_data()
        .try_to_vec::<f32>()?;
    let losses: Vec<_> = logits
        .chunks_exact(c.max_length * 5)
        .zip(&windows)
        .map(|(x, w)| {
            let result = deepconsensus_rs::alignment::alignment_loss(
                x,
                w.label.as_ref().unwrap_or(&w.ccs),
                c,
            )?;
            Ok(serde_json::json!({"loss":result.loss,"gradient":result.gradient}))
        })
        .collect::<Result<_>>()?;
    std::fs::write(
        &args[3],
        serde_json::to_vec(
            &serde_json::json!({"windows":windows,"logits":logits,"losses":losses}),
        )?,
    )?;
    Ok(())
}
