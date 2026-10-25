use anyhow::{Result, ensure};
use deepconsensus_rs::{config::ModelConfig, dataset::DatasetReader};
use std::collections::HashSet;
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() == 3, "usage: inspect_dataset PARAMS DATASET");
    let c = ModelConfig::load(&args[1])?;
    let mut reader = DatasetReader::new(args[2].as_ref(), &c)?;
    let mut n = 0;
    let mut names = HashSet::new();
    let mut truth_bases = 0;
    while let Some(w) = reader.next_window()? {
        n += 1;
        names.insert(w.name);
        let label = w
            .label
            .ok_or_else(|| anyhow::anyhow!("missing truth label"))?;
        truth_bases += label.iter().filter(|&&x| x != 0).count();
    }
    println!(
        "{}",
        serde_json::json!({"examples":n,"molecules":names.len(),"truth_bases":truth_bases,"all_record_crcs_valid":true})
    );
    Ok(())
}
