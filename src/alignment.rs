use crate::config::ModelConfig;
use anyhow::{Result, ensure};

#[derive(Debug)]
pub struct LossResult {
    pub loss: f64,
    pub gradient: Vec<f32>,
    pub active_cells: usize,
}

struct Row {
    lo: usize,
    values: Vec<f64>,
    adjoint: Vec<f64>,
}
impl Row {
    fn get(&self, j: usize) -> f64 {
        if j < self.lo {
            f64::INFINITY
        } else {
            self.values
                .get(j - self.lo)
                .copied()
                .unwrap_or(f64::INFINITY)
        }
    }
    fn add(&mut self, j: usize, v: f64) {
        if j >= self.lo
            && let Some(x) = self.adjoint.get_mut(j - self.lo)
        {
            *x += v;
        }
    }
}

fn bounds(i: usize, m: usize, n: usize, width: Option<usize>) -> (usize, usize) {
    if let Some(w) = width {
        let diagonal = i as f64;
        let endpoint = i as f64 * n as f64 / m.max(1) as f64;
        let lo = (diagonal.min(endpoint) - w as f64).ceil().max(0.0) as usize;
        let hi = (diagonal.max(endpoint) + w as f64).floor().min(n as f64) as usize;
        (lo.min(n + 1), hi)
    } else {
        (0, n)
    }
}

fn minimum(costs: [f64; 3], tau: Option<f64>) -> (f64, [f64; 3]) {
    let best = costs.into_iter().fold(f64::INFINITY, f64::min);
    if !best.is_finite() {
        return (f64::INFINITY, [0.0; 3]);
    }
    if let Some(t) = tau {
        let e = costs.map(|c| (-(c - best) / t).exp());
        let sum = e.iter().sum::<f64>();
        (best - t * sum.ln(), e.map(|v| v / sum))
    } else {
        let count = costs.iter().filter(|&&v| v == best).count() as f64;
        (
            best,
            costs.map(|v| if v == best { 1.0 / count } else { 0.0 }),
        )
    }
}

pub fn softmax(logits: &[f32]) -> Result<Vec<f64>> {
    ensure!(
        logits.len().is_multiple_of(5) && logits.iter().all(|x| x.is_finite()),
        "invalid logits"
    );
    let mut out = Vec::with_capacity(logits.len());
    for row in logits.as_chunks::<5>().0 {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let e: Vec<f64> = row.iter().map(|&v| (v as f64 - max).exp()).collect();
        let sum = e.iter().sum::<f64>();
        out.extend(e.iter().map(|v| v / sum));
    }
    Ok(out)
}

pub fn alignment_loss(logits: &[f32], label: &[u8], c: &ModelConfig) -> Result<LossResult> {
    ensure!(label.iter().all(|&x| x < 5), "invalid label token");
    let label: Vec<u8> = label.iter().copied().filter(|&x| x != 0).collect();
    let m = label.len();
    let n = logits.len() / 5;
    ensure!(
        n == c.max_length && m <= n,
        "alignment dimensions incompatible"
    );
    let p = softmax(logits)?;
    let eps = 1e-7;
    let cost = |j: usize, t: usize| -p[j * 5 + t].clamp(eps, 1.0 - eps).ln();
    let mut grid = Vec::<Row>::with_capacity(m + 1);
    let mut cells = 0;
    // Store and evaluate only cells in the corridor, rather than dense costs plus a mask.
    for i in 0..=m {
        let (lo, hi) = bounds(i, m, n, c.band_width);
        let len = if hi >= lo { hi - lo + 1 } else { 0 };
        cells += len;
        let mut row = Row {
            lo,
            values: vec![f64::INFINITY; len],
            adjoint: vec![0.0; len],
        };
        for j in lo..lo + len {
            let v = if i == 0 && j == 0 {
                0.0
            } else {
                let sub = if i > 0 && j > 0 {
                    grid[i - 1].get(j - 1) + cost(j - 1, label[i - 1] as usize)
                } else {
                    f64::INFINITY
                };
                let ins = if j > 0 {
                    row.get(j - 1) + cost(j - 1, 0)
                } else {
                    f64::INFINITY
                };
                let del = if i > 0 {
                    grid[i - 1].get(j) + c.del_cost
                } else {
                    f64::INFINITY
                };
                minimum([sub, ins, del], c.loss_reg).0
            };
            row.values[j - lo] = v;
        }
        grid.push(row);
    }
    let loss = grid[m].get(n);
    ensure!(
        loss.is_finite(),
        "band excludes all complete alignment paths; increase band_width"
    );
    grid[m].add(n, 1.0);
    let mut dp = vec![0f64; p.len()];
    for i in (0..=m).rev() {
        let lo = grid[i].lo;
        let hi = lo + grid[i].values.len();
        for j in (lo..hi).rev() {
            if i == 0 && j == 0 {
                continue;
            }
            let a = grid[i].adjoint[j - lo];
            if a == 0.0 {
                continue;
            }
            let sub = if i > 0 && j > 0 {
                grid[i - 1].get(j - 1) + cost(j - 1, label[i - 1] as usize)
            } else {
                f64::INFINITY
            };
            let ins = if j > 0 {
                grid[i].get(j - 1) + cost(j - 1, 0)
            } else {
                f64::INFINITY
            };
            let del = if i > 0 {
                grid[i - 1].get(j) + c.del_cost
            } else {
                f64::INFINITY
            };
            let weights = minimum([sub, ins, del], c.loss_reg).1.map(|w| a * w);
            if i > 0 && j > 0 {
                grid[i - 1].add(j - 1, weights[0]);
                let k = (j - 1) * 5 + label[i - 1] as usize;
                if p[k] >= eps && p[k] <= 1.0 - eps {
                    dp[k] -= weights[0] / p[k];
                }
            }
            if j > 0 {
                grid[i].add(j - 1, weights[1]);
                let k = (j - 1) * 5;
                if p[k] >= eps && p[k] <= 1.0 - eps {
                    dp[k] -= weights[1] / p[k];
                }
            }
            if i > 0 {
                grid[i - 1].add(j, weights[2]);
            }
        }
    }
    let mut gradient = vec![0f32; p.len()];
    for j in 0..n {
        let dot = (0..5).map(|k| dp[j * 5 + k] * p[j * 5 + k]).sum::<f64>();
        for k in 0..5 {
            gradient[j * 5 + k] = (p[j * 5 + k] * (dp[j * 5 + k] - dot)) as f32;
        }
    }
    Ok(LossResult {
        loss,
        gradient,
        active_cells: cells,
    })
}

pub fn wavefront_loss(
    logits: burn::tensor::Tensor<3>,
    labels: &[Vec<u8>],
    c: &ModelConfig,
) -> Result<burn::tensor::Tensor<1>> {
    use burn::tensor::{Bool, Int, Tensor, TensorData, activation};
    let [b, n, k] = logits.dims();
    ensure!(
        b == labels.len() && n == c.max_length && k == 5,
        "wavefront batch dimensions mismatch"
    );
    let labels: Vec<Vec<u8>> = labels
        .iter()
        .map(|v| v.iter().copied().filter(|&x| x != 0).collect())
        .collect();
    ensure!(
        labels
            .iter()
            .all(|l| l.len() <= n && l.iter().all(|&x| x < 5)),
        "invalid wavefront labels"
    );
    let device = logits.device();
    let logp = activation::softmax(logits, 2)
        .clamp(1e-7, 1.0 - 1e-7)
        .log()
        .reshape([b, n * 5]);
    let inf = || Tensor::<2>::full([b, n + 1], 1e9, &device);
    let mut initial = vec![1e9f32; b * (n + 1)];
    for row in 0..b {
        initial[row * (n + 1)] = 0.0;
    }
    let mut previous = Tensor::<2>::from_data(TensorData::new(initial, [b, n + 1]), &device);
    let mut previous2 = inf();
    let lengths = Tensor::<2, Int>::from_data(
        TensorData::new(
            labels.iter().map(|l| l.len() as i64).collect::<Vec<_>>(),
            [b, 1],
        ),
        &device,
    );
    let mut output = Tensor::<2>::full([b, 1], 1e9, &device);
    let shift = |x: Tensor<2>| {
        Tensor::cat(
            vec![
                Tensor::<2>::full([b, 1], 1e9, &device),
                x.slice([0..b, 0..n]),
            ],
            1,
        )
    };
    // Each anti-diagonal is a batched device operation on CUDA or Metal with autodiff.
    for diagonal in 1..=2 * n {
        let mut subs = Vec::with_capacity(b * (n + 1));
        let mut ins = Vec::with_capacity(b * (n + 1));
        let mut masked = Vec::with_capacity(b * (n + 1));
        let mut endpoint = Vec::with_capacity(b);
        for label in &labels {
            endpoint.push(diagonal == label.len() + n);
            for i in 0..=n {
                let j = diagonal as isize - i as isize;
                let pos = (j - 1).clamp(0, n as isize - 1) as usize;
                subs.push(
                    (pos * 5 + label.get(i.wrapping_sub(1)).copied().unwrap_or(0) as usize) as i64,
                );
                ins.push((pos * 5) as i64);
                let (lo, hi) = bounds(i, label.len(), n, c.band_width);
                masked.push(
                    j < 0
                        || j > n as isize
                        || i > label.len()
                        || j < lo as isize
                        || j > hi as isize,
                );
            }
        }
        let sub_index = Tensor::<2, Int>::from_data(TensorData::new(subs, [b, n + 1]), &device);
        let ins_index = Tensor::<2, Int>::from_data(TensorData::new(ins, [b, n + 1]), &device);
        let sub = shift(previous2) - logp.clone().gather(1, sub_index);
        let insert = previous.clone() - logp.clone().gather(1, ins_index);
        let delete = shift(previous.clone()) + c.del_cost;
        let candidates: Tensor<3> = Tensor::stack(vec![sub, insert, delete], 2);
        let best = candidates.clone().min_dim(2).detach();
        let current = if let Some(tau) = c.loss_reg {
            best.clone() - ((candidates - best) / (-tau)).exp().sum_dim(2).log() * tau
        } else {
            candidates.min_dim(2)
        };
        let mask = Tensor::<2, Bool>::from_data(TensorData::new(masked, [b, n + 1]), &device);
        let current = current.reshape([b, n + 1]).mask_fill(mask, 1e9);
        let done = Tensor::<2, Bool>::from_data(TensorData::new(endpoint, [b, 1]), &device);
        output = output.mask_where(done, current.clone().gather(1, lengths.clone()));
        previous2 = previous;
        previous = current;
    }
    Ok(output.mean())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn device_wavefront_matches_analytic_vjp() {
        use burn::tensor::{Device, Tensor, TensorData};
        let device = Device::autodiff(crate::model::device("cpu").unwrap());
        for width in [None, Some(2)] {
            let c = ModelConfig {
                max_length: 4,
                loss_reg: Some(0.3),
                band_width: width,
                del_cost: 2.0,
                ..Default::default()
            };
            let values: Vec<f32> = (0..40).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.2).collect();
            let labels = vec![vec![1, 0, 2, 3], vec![2, 4]];
            let x = Tensor::<3>::from_data(TensorData::new(values.clone(), [2, 4, 5]), &device)
                .require_grad();
            let loss = wavefront_loss(x.clone(), &labels, &c).unwrap();
            let value = loss.clone().into_scalar::<f32>() as f64;
            let grads = loss.backward();
            let gradient = x
                .grad(&grads)
                .unwrap()
                .to_data()
                .try_to_vec::<f32>()
                .unwrap();
            let a = alignment_loss(&values[..20], &labels[0], &c).unwrap();
            let b = alignment_loss(&values[20..], &labels[1], &c).unwrap();
            assert!((value - (a.loss + b.loss) / 2.0).abs() < 1e-5);
            for (&g, expected) in gradient.iter().zip(a.gradient.iter().chain(&b.gradient)) {
                assert!((g - expected / 2.0).abs() < 1e-5, "{g} vs {expected}");
            }
        }
    }
    #[test]
    fn gradients_match_finite_difference() {
        for width in [None, Some(2)] {
            let c = ModelConfig {
                max_length: 4,
                band_width: width,
                loss_reg: Some(0.3),
                del_cost: 2.0,
                ..Default::default()
            };
            let x: Vec<f32> = (0..20).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.3).collect();
            let label = [1, 2, 0, 3];
            let r = alignment_loss(&x, &label, &c).unwrap();
            for k in 0..20 {
                let mut a = x.clone();
                let mut b = x.clone();
                a[k] += 0.001;
                b[k] -= 0.001;
                let numeric = (alignment_loss(&a, &label, &c).unwrap().loss
                    - alignment_loss(&b, &label, &c).unwrap().loss)
                    / 0.002;
                assert!(
                    (numeric - r.gradient[k] as f64).abs() < 0.001,
                    "{k}: {numeric} != {}",
                    r.gradient[k]
                );
            }
        }
    }
    #[test]
    fn wide_band_is_dense() {
        let c = ModelConfig {
            max_length: 4,
            ..Default::default()
        };
        let x = vec![0.0; 20];
        let a = alignment_loss(&x, &[1, 2], &c).unwrap();
        let b = alignment_loss(
            &x,
            &[1, 2],
            &ModelConfig {
                band_width: Some(4),
                ..c
            },
        )
        .unwrap();
        assert_eq!(a.loss, b.loss);
        assert_eq!(a.gradient, b.gradient);
    }
    #[test]
    fn endpoint_included() {
        assert_eq!(bounds(80, 80, 100, Some(5)), (75, 100));
    }
    #[test]
    fn empty_truth_and_clipping() {
        let c = ModelConfig {
            max_length: 3,
            ..Default::default()
        };
        let r = alignment_loss(&[100.0, 0.0, 0.0, 0.0, 0.0].repeat(3), &[], &c).unwrap();
        assert!(r.loss.is_finite());
        assert!(r.gradient.iter().all(|x| x.is_finite()));
    }
}
