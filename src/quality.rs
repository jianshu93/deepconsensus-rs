use anyhow::{Result, ensure};

#[derive(Clone, Copy, Debug)]
pub struct Calibration {
    pub threshold: f64,
    pub slope: f64,
    pub intercept: f64,
}

impl Calibration {
    pub fn parse(value: &str) -> Result<Self> {
        if value == "skip" {
            return Ok(Self {
                threshold: 0.0,
                slope: 1.0,
                intercept: 0.0,
            });
        }
        let v: Vec<f64> = value
            .split(',')
            .map(str::parse)
            .collect::<std::result::Result<_, _>>()?;
        ensure!(
            v.len() == 3 && v.iter().all(|x| x.is_finite()),
            "calibration must be skip or threshold,slope,intercept"
        );
        Ok(Self {
            threshold: v[0],
            slope: v[1],
            intercept: v[2],
        })
    }
    pub fn apply(self, q: f64, max: u8) -> u8 {
        let calibrated = if self.threshold == 0.0 || q > self.threshold {
            self.slope * q + self.intercept
        } else {
            q
        };
        calibrated.clamp(0.0, max as f64).round_ties_even() as u8
    }
}

pub fn avg_phred(qualities: impl IntoIterator<Item = i16>) -> f64 {
    let mut sum = 0.0;
    let mut n = 0;
    for q in qualities {
        if q >= 0 {
            sum += 10f64.powf(-(q as f64) / 10.0);
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        -10.0 * (sum / n as f64).log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn probability_average() {
        assert!((avg_phred([10, 30, -1]) - 12.967086).abs() < 1e-5);
    }
    #[test]
    fn calibration_threshold() {
        let c = Calibration::parse("20,0.5,2").unwrap();
        assert_eq!(c.apply(20.0, 93), 20);
        assert_eq!(c.apply(30.0, 93), 17);
    }
}
