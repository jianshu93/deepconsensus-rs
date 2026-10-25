use crate::config::ModelConfig;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

pub const VOCAB: &[u8; 5] = b" ATCG";
pub fn encode(base: u8) -> Result<u8> {
    match base.to_ascii_uppercase() {
        b' ' | b'-' => Ok(0),
        b'A' => Ok(1),
        b'T' => Ok(2),
        b'C' => Ok(3),
        b'G' => Ok(4),
        _ => anyhow::bail!(
            "unsupported base {:?}; do not silently map ambiguity to a trained token",
            base as char
        ),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlignedRead {
    pub bases: Vec<u8>,
    pub insertion: Vec<bool>,
    pub pw: Vec<u8>,
    pub ip: Vec<u8>,
    pub ccs_index: Vec<i32>,
    pub strand: u8,
    pub sn: [f32; 4],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Molecule {
    pub name: String,
    pub subreads: Vec<AlignedRead>,
    pub ccs: Vec<u8>,
    pub qualities: Vec<u8>,
    pub smart_windows: Option<Vec<usize>>,
    pub truth: Option<AlignedRead>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Window {
    pub name: String,
    pub window_pos: usize,
    pub length: usize,
    pub passes: usize,
    pub rows: Vec<f32>,
    pub label: Option<Vec<u8>>,
    pub ccs: Vec<u8>,
    pub qualities: Vec<i16>,
    pub overflow: bool,
}

impl AlignedRead {
    pub fn validate(&self) -> Result<()> {
        let n = self.bases.len();
        ensure!(
            self.insertion.len() == n
                && self.pw.len() == n
                && self.ip.len() == n
                && self.ccs_index.len() == n,
            "inconsistent expanded read lengths"
        );
        ensure!(
            self.bases.iter().all(|&x| x < 5)
                && self.strand <= 2
                && self.sn.iter().all(|x| x.is_finite()),
            "invalid expanded read"
        );
        Ok(())
    }
}

pub fn windows(m: &Molecule, config: &ModelConfig, smart: bool) -> Result<Vec<Window>> {
    windows_with_gate(m, config, smart, None)
}

pub fn windows_with_gate(
    m: &Molecule,
    config: &ModelConfig,
    smart: bool,
    skip_above: Option<f64>,
) -> Result<Vec<Window>> {
    config.validate()?;
    ensure!(
        !m.subreads.is_empty() && !m.ccs.is_empty(),
        "molecule {} has no aligned evidence",
        m.name
    );
    ensure!(
        m.ccs.len() == m.qualities.len(),
        "CCS quality length mismatch"
    );
    ensure!(m.ccs.iter().all(|&x| x < 5), "invalid CCS tokens");
    for r in &m.subreads {
        r.validate()?;
    }
    // Layout uses ALL passes, including those not selected as model feature rows.
    let ccs_read = AlignedRead {
        bases: m.ccs.clone(),
        insertion: vec![false; m.ccs.len()],
        pw: vec![0; m.ccs.len()],
        ip: vec![0; m.ccs.len()],
        ccs_index: (0..m.ccs.len() as i32).collect(),
        strand: 0,
        sn: [0.0; 4],
    };
    let passes = m.subreads.len().min(config.max_passes);
    let all: Vec<&AlignedRead> = m
        .subreads
        .iter()
        .chain(std::iter::once(&ccs_read))
        .collect();
    let mapped = event_layout(&all, passes)?;
    let layout = &mapped[passes];
    let mut widths = Vec::new();
    if smart {
        let requested = m
            .smart_windows
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("smart window tag missing for {}", m.name))?;
        ensure!(
            requested.iter().all(|&w| w > 0) && requested.iter().sum::<usize>() == m.ccs.len(),
            "invalid smart-window coverage"
        );
        let mut p = 0;
        for &width in requested {
            let start = p;
            let mut consumed = 0;
            while consumed < width {
                ensure!(p < layout.len(), "smart windows exceed layout");
                consumed += usize::from(layout[p] != u32::MAX);
                p += 1;
            }
            widths.push(p - start);
        }
        if p < layout.len() {
            *widths.last_mut().unwrap() += layout.len() - p;
        }
    } else {
        widths = vec![config.max_length; layout.len().div_ceil(config.max_length)];
    }
    let mut out = Vec::new();
    let mut start = 0;
    for width in widths {
        let end = (start + width).min(layout.len());
        let indices: Vec<usize> = layout[start..end]
            .iter()
            .copied()
            .filter(|&x| x != u32::MAX)
            .map(|x| x as usize)
            .collect();
        if indices.is_empty() {
            start += width;
            continue;
        }
        let n = if width > config.max_length {
            end - start
        } else {
            config.max_length
        };
        let mut ccs = vec![0; n];
        let mut qualities = vec![-1; n];
        for t in 0..n {
            if start + t < end && layout[start + t] != u32::MAX {
                let k = layout[start + t] as usize;
                ccs[t] = m.ccs[k];
                qualities[t] = m.qualities[k] as i16;
            }
        }
        let skip_features = skip_above.is_some_and(|threshold| {
            width > config.max_length
                || (threshold > 0.0
                    && crate::quality::avg_phred(qualities.iter().copied()) > threshold)
        });
        let mut rows = if skip_features {
            Vec::new()
        } else {
            vec![0f32; config.rows() * n]
        };
        if !skip_features {
            for r in 0..passes {
                for t in 0..n {
                    if let Some(&k) = mapped[r]
                        .get(start + t)
                        .filter(|&&k| start + t < end && k != u32::MAX)
                    {
                        rows[r * n + t] = m.subreads[r].bases[k as usize] as f32;
                        rows[(config.max_passes + r) * n + t] = m.subreads[r].pw[k as usize] as f32;
                        rows[(2 * config.max_passes + r) * n + t] =
                            m.subreads[r].ip[k as usize] as f32;
                    }
                    rows[(3 * config.max_passes + r) * n + t] = m.subreads[r].strand as f32;
                }
            }
            for t in 0..n {
                rows[4 * config.max_passes * n + t] = ccs[t] as f32;
                if config.use_ccs_bq {
                    rows[(4 * config.max_passes + 1) * n + t] = qualities[t] as f32;
                }
                for s in 0..4 {
                    rows[(config.rows() - 4 + s) * n + t] = m.subreads[0].sn[s];
                }
            }
        }
        // Truth insertions must not add columns to the model input. Include them in the label slice.
        let label = if let Some(truth) = &m.truth {
            truth.validate()?;
            let lo = *indices.first().unwrap() as i32;
            let hi = *indices.last().unwrap() as i32;
            let selected: Vec<usize> = truth
                .ccs_index
                .iter()
                .enumerate()
                .filter_map(|(k, &i)| (i >= lo && i <= hi).then_some(k))
                .collect();
            if selected.is_empty() {
                start += width;
                continue;
            }
            let left = *selected.first().unwrap();
            let right = *selected.last().unwrap() + 1;
            let mut label: Vec<u8> = truth.bases[left..right]
                .iter()
                .copied()
                .filter(|&b| b != 0)
                .collect();
            if label.len() > config.max_length || width > config.max_length {
                start += width;
                continue;
            }
            label.resize(config.max_length, 0);
            Some(label)
        } else {
            None
        };
        out.push(Window {
            name: m.name.clone(),
            window_pos: indices[0],
            length: n,
            passes,
            rows,
            label,
            ccs,
            qualities,
            overflow: width > config.max_length,
        });
        start += width;
    }
    Ok(out)
}

fn event_layout(reads: &[&AlignedRead], selected: usize) -> Result<Vec<Vec<u32>>> {
    let reference_len = reads
        .iter()
        .map(|r| r.insertion.iter().filter(|&&x| !x).count())
        .max()
        .unwrap_or(0);
    let mut insertions = vec![0usize; reference_len + 1];
    for read in reads {
        ensure!(
            read.bases.len() < u32::MAX as usize,
            "read exceeds compact index limit"
        );
        let mut reference = 0;
        let mut k = 0;
        while k < read.bases.len() {
            if read.insertion[k] {
                let begin = k;
                while k < read.bases.len() && read.insertion[k] {
                    k += 1;
                }
                insertions[reference] = insertions[reference].max(k - begin);
            } else {
                reference += 1;
                k += 1;
            }
        }
    }
    let mut offsets = Vec::with_capacity(reference_len + 1);
    let mut total = 0;
    for (i, &count) in insertions.iter().enumerate() {
        offsets.push(total);
        total += count + usize::from(i < reference_len);
    }
    let mut out = Vec::with_capacity(selected + 1);
    // Only selected feature passes need a materialized map; all passes still set insertion widths.
    for read in reads
        .iter()
        .take(selected)
        .copied()
        .chain(reads.last().copied())
    {
        let mut map = vec![u32::MAX; total];
        let mut reference = 0;
        let mut insertion = 0;
        for (k, &is_insertion) in read.insertion.iter().enumerate() {
            if is_insertion {
                map[offsets[reference] + insertion] = k as u32;
                insertion += 1;
            } else {
                map[offsets[reference] + insertions[reference]] = k as u32;
                reference += 1;
                insertion = 0;
            }
        }
        out.push(map);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn event_layout_matches_column_walk() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(81);
        for _ in 0..200 {
            let reads: Vec<_> = (0..rng.gen_range(2..25))
                .map(|_| {
                    let n = rng.gen_range(1..60);
                    AlignedRead {
                        bases: vec![1; n],
                        insertion: (0..n).map(|_| rng.gen_bool(0.3)).collect(),
                        pw: vec![0; n],
                        ip: vec![0; n],
                        ccs_index: vec![-1; n],
                        strand: 1,
                        sn: [1.0; 4],
                    }
                })
                .collect();
            let refs: Vec<_> = reads.iter().collect();
            let selected = refs.len() - 1;
            let actual = event_layout(&refs, selected).unwrap();
            let mut cursor = vec![0; refs.len()];
            let mut expected = vec![Vec::new(); refs.len()];
            while refs
                .iter()
                .enumerate()
                .any(|(i, r)| cursor[i] < r.bases.len())
            {
                let insertion = refs
                    .iter()
                    .enumerate()
                    .any(|(i, r)| cursor[i] < r.bases.len() && r.insertion[cursor[i]]);
                for (i, r) in refs.iter().enumerate() {
                    if cursor[i] >= r.bases.len() || (insertion && !r.insertion[cursor[i]]) {
                        expected[i].push(u32::MAX);
                    } else {
                        expected[i].push(cursor[i] as u32);
                        cursor[i] += 1;
                    }
                }
            }
            assert_eq!(actual, expected);
        }
    }
    #[test]
    fn unselected_pass_changes_columns() {
        let r = AlignedRead {
            bases: vec![1, 2],
            insertion: vec![false, false],
            pw: vec![8, 9],
            ip: vec![4, 5],
            ccs_index: vec![0, 1],
            strand: 1,
            sn: [1.0; 4],
        };
        let mut r2 = r.clone();
        r2.bases = vec![1, 3, 2];
        r2.insertion = vec![false, true, false];
        r2.pw = vec![0; 3];
        r2.ip = vec![0; 3];
        r2.ccs_index = vec![0, -1, 1];
        let m = Molecule {
            name: "m/1/ccs".into(),
            subreads: vec![r, r2],
            ccs: vec![1, 2],
            qualities: vec![30, 31],
            smart_windows: None,
            truth: None,
        };
        let c = ModelConfig {
            max_passes: 1,
            max_length: 4,
            ..Default::default()
        };
        let w = windows(&m, &c, false).unwrap();
        assert_eq!(w[0].ccs, vec![1, 0, 2, 0]);
        assert_eq!(w[0].qualities, vec![30, -1, 31, -1]);
        assert_eq!(&w[0].rows[..4], &[1.0, 0.0, 2.0, 0.0]);
        let skipped = windows_with_gate(&m, &c, false, Some(20.0)).unwrap();
        assert!(skipped[0].rows.is_empty());
        assert_eq!(skipped[0].ccs, w[0].ccs);
        assert_eq!(skipped[0].qualities, w[0].qualities);
        let mut smart = m.clone();
        smart.smart_windows = Some(vec![1, 1]);
        let smart_windows = windows(&smart, &c, true).unwrap();
        assert_eq!(smart_windows.len(), 2);
        assert_eq!(smart_windows[0].window_pos, 0);
        assert_eq!(smart_windows[1].window_pos, 1);
        assert_eq!(smart_windows[1].ccs, vec![0, 2, 0, 0]);
        smart.smart_windows = Some(vec![2]);
        let overflow = windows(
            &smart,
            &ModelConfig {
                max_length: 2,
                ..c.clone()
            },
            true,
        )
        .unwrap();
        assert!(overflow[0].overflow);
        assert_eq!(overflow[0].ccs, vec![1, 0, 2]);
    }
}
