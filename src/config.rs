use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    pub max_length: usize,
    pub max_passes: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_hidden_layers: usize,
    pub filter_size: usize,
    pub attn_win_size: Option<usize>,
    pub per_base_hidden_size: usize,
    pub pw_hidden_size: usize,
    pub ip_hidden_size: usize,
    pub strand_hidden_size: usize,
    pub ccs_bq_hidden_size: usize,
    pub sn_hidden_size: usize,
    pub use_ccs_bq: bool,
    pub rezero: bool,
    pub add_pos_encoding: bool,
    pub attention_dropout: f64,
    pub relu_dropout: f64,
    pub layer_postprocess_dropout: f64,
    pub del_cost: f64,
    pub loss_reg: Option<f64>,
    pub band_width: Option<usize>,
    pub dc_calibration: String,
    #[serde(rename = "PW_MAX")]
    pub pw_max: usize,
    #[serde(rename = "IP_MAX")]
    pub ip_max: usize,
    #[serde(rename = "SN_MAX")]
    pub sn_max: usize,
    #[serde(rename = "CCS_BQ_MAX")]
    pub ccs_bq_max: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            max_length: 100,
            max_passes: 20,
            hidden_size: 280,
            num_heads: 2,
            num_hidden_layers: 5,
            filter_size: 2048,
            attn_win_size: Some(12),
            per_base_hidden_size: 8,
            pw_hidden_size: 8,
            ip_hidden_size: 8,
            strand_hidden_size: 2,
            ccs_bq_hidden_size: 8,
            sn_hidden_size: 8,
            use_ccs_bq: true,
            rezero: true,
            add_pos_encoding: true,
            attention_dropout: 0.1,
            relu_dropout: 0.0,
            layer_postprocess_dropout: 0.0,
            del_cost: 10.0,
            loss_reg: Some(0.1),
            band_width: None,
            dc_calibration: "skip".into(),
            pw_max: 255,
            ip_max: 255,
            sn_max: 500,
            ccs_bq_max: 95,
        }
    }
}

impl ModelConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_reader(std::fs::File::open(path)?)?;
        for feature in [
            "use_bases",
            "use_pw",
            "use_ip",
            "use_strand",
            "use_ccs",
            "use_sn",
            "condense_transformer_input",
        ] {
            ensure!(
                value.get(feature).is_none_or(|v| v.as_bool() == Some(true)),
                "unsupported architecture: {feature} must be true"
            );
        }
        for feature in ["use_dnabert", "layer_norm"] {
            ensure!(
                value
                    .get(feature)
                    .is_none_or(|v| v.as_bool() == Some(false)),
                "unsupported architecture option {feature}"
            );
        }
        if let Some(name) = value.get("model_name").and_then(|v| v.as_str()) {
            ensure!(
                matches!(
                    name,
                    "transformer_learn_values" | "transformer_learn_values_distill"
                ),
                "unsupported architecture {name}"
            );
        }
        let config: Self = serde_json::from_value(value)?;
        config.validate()?;
        Ok(config)
    }
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        std::fs::write(path, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }
    pub fn rows(&self) -> usize {
        4 * self.max_passes + 5 + usize::from(self.use_ccs_bq)
    }
    pub fn embedding_width(&self) -> usize {
        self.max_passes
            * (self.per_base_hidden_size
                + self.pw_hidden_size
                + self.ip_hidden_size
                + self.strand_hidden_size)
            + self.per_base_hidden_size
            + 4 * self.sn_hidden_size
            + if self.use_ccs_bq {
                self.ccs_bq_hidden_size
            } else {
                0
            }
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.max_length > 0 && self.max_passes > 0,
            "window length and passes must be positive"
        );
        ensure!(
            self.num_heads > 0
                && self.hidden_size >= 4
                && self.hidden_size.is_multiple_of(self.num_heads),
            "hidden_size must be divisible by num_heads"
        );
        ensure!(
            !self.add_pos_encoding || self.hidden_size.is_multiple_of(2),
            "TF positional encoding requires an even hidden size"
        );
        ensure!(
            self.num_hidden_layers > 0 && self.filter_size > 0,
            "empty encoder"
        );
        ensure!(
            [
                self.per_base_hidden_size,
                self.pw_hidden_size,
                self.ip_hidden_size,
                self.strand_hidden_size,
                self.ccs_bq_hidden_size,
                self.sn_hidden_size
            ]
            .iter()
            .all(|&x| x > 0),
            "embedding widths must be positive"
        );
        ensure!(
            self.del_cost.is_finite() && self.del_cost >= 0.0,
            "invalid deletion cost"
        );
        ensure!(
            self.loss_reg.is_none_or(|v| v.is_finite() && v > 0.0),
            "loss_reg must be positive or null"
        );
        for p in [
            self.attention_dropout,
            self.relu_dropout,
            self.layer_postprocess_dropout,
        ] {
            ensure!(p.is_finite() && (0.0..1.0).contains(&p), "invalid dropout");
        }
        ensure!(
            self.ccs_bq_max >= 95 && self.pw_max > 0 && self.ip_max > 0 && self.sn_max > 0,
            "invalid feature vocabularies"
        );
        Ok(())
    }
}
