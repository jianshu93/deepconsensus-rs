use crate::preprocess::{AlignedRead, Molecule, encode};
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use rust_htslib::bam::{
    self, Read, Record,
    record::{Aux, Cigar},
};
use std::path::Path;

fn next(reader: &mut bam::Reader) -> Result<Option<Record>> {
    let mut r = Record::new();
    match reader.read(&mut r) {
        Some(x) => {
            x?;
            Ok(Some(r))
        }
        None => Ok(None),
    }
}

fn numeric_array(record: &Record, tag: &[u8; 2]) -> Result<Vec<f64>> {
    let aux = record.aux(tag).with_context(|| {
        format!(
            "missing {} tag for {}",
            String::from_utf8_lossy(tag),
            String::from_utf8_lossy(record.qname())
        )
    })?;
    Ok(match aux {
        Aux::ArrayU8(a) => a.iter().map(|v| v as f64).collect(),
        Aux::ArrayU16(a) => a.iter().map(|v| v as f64).collect(),
        Aux::ArrayU32(a) => a.iter().map(|v| v as f64).collect(),
        Aux::ArrayI8(a) => a.iter().map(|v| v as f64).collect(),
        Aux::ArrayI16(a) => a.iter().map(|v| v as f64).collect(),
        Aux::ArrayI32(a) => a.iter().map(|v| v as f64).collect(),
        Aux::ArrayFloat(a) => a.iter().map(|v| v as f64).collect(),
        _ => anyhow::bail!(
            "tag {} is not a numeric array",
            String::from_utf8_lossy(tag)
        ),
    })
}

pub fn expand(record: &Record, truth: bool, ins_trim: usize) -> Result<AlignedRead> {
    ensure!(
        !record.is_unmapped() && record.pos() >= 0,
        "expected mapped read"
    );
    let seq = record.seq().as_bytes();
    let seq: Vec<u8> = seq.into_iter().map(encode).collect::<Result<_>>()?;
    let mut pw = if truth {
        vec![0; seq.len()]
    } else {
        numeric_array(record, b"pw")?
            .into_iter()
            .map(|x| x as u32 as u8)
            .collect()
    };
    let mut ip = if truth {
        vec![0; seq.len()]
    } else {
        numeric_array(record, b"ip")?
            .into_iter()
            .map(|x| x as u32 as u8)
            .collect()
    };
    ensure!(
        pw.len() == seq.len() && ip.len() == seq.len(),
        "kinetic tag length mismatch"
    );
    let sn = if truth {
        [0.0; 4]
    } else {
        let sn = numeric_array(record, b"sn")?;
        ensure!(sn.len() == 4, "sn needs four values");
        [sn[0] as f32, sn[1] as f32, sn[2] as f32, sn[3] as f32]
    };
    if record.is_reverse() {
        pw.reverse();
        ip.reverse();
    }
    let mut out = AlignedRead {
        bases: vec![0; record.pos() as usize],
        insertion: vec![false; record.pos() as usize],
        pw: vec![0; record.pos() as usize],
        ip: vec![0; record.pos() as usize],
        ccs_index: vec![-1; record.pos() as usize],
        strand: if record.is_reverse() { 2 } else { 1 },
        sn,
    };
    let mut q = 0usize;
    let mut reference = record.pos() as i32;
    for op in record.cigar().iter() {
        let len = op.len() as usize;
        match op {
            Cigar::Match(_) | Cigar::Equal(_) | Cigar::Diff(_) => {
                for _ in 0..len {
                    ensure!(q < seq.len(), "CIGAR exceeds query");
                    out.bases.push(seq[q]);
                    out.insertion.push(false);
                    out.pw.push(pw[q]);
                    out.ip.push(ip[q]);
                    out.ccs_index.push(reference);
                    q += 1;
                    reference += 1;
                }
            }
            Cigar::Ins(_) => {
                let keep = ins_trim == 0 || len <= ins_trim;
                for _ in 0..len {
                    ensure!(q < seq.len(), "CIGAR exceeds query");
                    if keep {
                        out.bases.push(seq[q]);
                        out.insertion.push(true);
                        out.pw.push(pw[q]);
                        out.ip.push(ip[q]);
                        out.ccs_index.push(-1);
                    }
                    q += 1;
                }
            }
            Cigar::Del(_) | Cigar::RefSkip(_) => {
                for _ in 0..len {
                    out.bases.push(0);
                    out.insertion.push(false);
                    out.pw.push(0);
                    out.ip.push(0);
                    out.ccs_index.push(reference);
                    reference += 1;
                }
            }
            Cigar::SoftClip(_) => {
                q += len;
                ensure!(q <= seq.len(), "soft clip exceeds query");
            }
            Cigar::HardClip(_) => {}
            Cigar::Pad(_) => anyhow::bail!("padded CIGAR is not supported"),
        }
    }
    ensure!(
        q == seq.len(),
        "CIGAR consumes {q} of {} query bases",
        seq.len()
    );
    out.validate()?;
    Ok(out)
}

pub struct MoleculeReader {
    subreads: bam::Reader,
    ccs: bam::Reader,
    truth: Option<bam::IndexedReader>,
    pending: Option<Record>,
    ins_trim: usize,
}

impl MoleculeReader {
    pub fn open(
        subreads: impl AsRef<Path>,
        ccs: impl AsRef<Path>,
        truth: Option<&Path>,
        threads: usize,
        ins_trim: usize,
    ) -> Result<Self> {
        let mut subreads = bam::Reader::from_path(subreads)?;
        let mut ccs = bam::Reader::from_path(ccs)?;
        if threads > 0 {
            subreads.set_threads(threads)?;
            ccs.set_threads(threads)?;
        }
        let truth = truth.map(bam::IndexedReader::from_path).transpose()?;
        Ok(Self {
            subreads,
            ccs,
            truth,
            pending: None,
            ins_trim,
        })
    }
    pub fn next_molecule(&mut self) -> Result<Option<Molecule>> {
        loop {
            let first = loop {
                let record = match self.pending.take() {
                    Some(r) => Some(r),
                    None => next(&mut self.subreads)?,
                };
                match record {
                    Some(r) if !r.is_unmapped() => break r,
                    Some(_) => continue,
                    None => return Ok(None),
                }
            };
            let tid = first.tid();
            ensure!(tid >= 0, "mapped record has no reference");
            let name = String::from_utf8(self.subreads.header().tid2name(tid as u32).to_vec())?;
            let mut records = vec![first];
            while let Some(r) = next(&mut self.subreads)? {
                if r.is_unmapped() {
                    continue;
                }
                if r.tid() != tid {
                    self.pending = Some(r);
                    break;
                }
                records.push(r);
            }
            let ccs = loop {
                let r = next(&mut self.ccs)?.ok_or_else(|| {
                    anyhow::anyhow!("CCS {name} missing or BAMs ordered differently")
                })?;
                if r.qname() == name.as_bytes() {
                    break r;
                }
            };
            ensure!(
                ccs.qual().iter().all(|&x| x <= 93),
                "missing or invalid CCS qualities for {name}"
            );
            let sequence: Vec<u8> = ccs
                .seq()
                .as_bytes()
                .into_iter()
                .map(encode)
                .collect::<Result<_>>()?;
            let smart_windows = ccs
                .aux(b"wl")
                .ok()
                .map(|_| {
                    numeric_array(&ccs, b"wl").map(|v| v.into_iter().map(|x| x as usize).collect())
                })
                .transpose()?;
            let truth = if let Some(reader) = self.truth.as_mut() {
                let truth_tid = reader
                    .header()
                    .tid(name.as_bytes())
                    .ok_or_else(|| anyhow::anyhow!("truth BAM has no reference {name}"))?;
                reader.fetch((truth_tid, 0, sequence.len() as i64))?;
                let mut aligned = Vec::new();
                for r in reader.records() {
                    let r = r?;
                    if !r.is_unmapped() && !r.is_secondary() {
                        aligned.push(r);
                    }
                }
                if aligned.is_empty() {
                    continue;
                }
                ensure!(
                    aligned.len() == 1 && !aligned[0].is_supplementary(),
                    "ambiguous/supplementary truth alignment for {name}"
                );
                Some(expand(&aligned[0], true, 0)?)
            } else {
                None
            };
            let subreads = records
                .par_iter()
                .map(|r| expand(r, false, self.ins_trim))
                .collect::<Result<_>>()?;
            return Ok(Some(Molecule {
                name,
                subreads,
                ccs: sequence,
                qualities: ccs.qual().to_vec(),
                smart_windows,
                truth,
            }));
        }
    }
}
