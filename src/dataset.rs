use crate::{config::ModelConfig, preprocess::Window};
use anyhow::{Result, ensure};
use flate2::{Compression, read::MultiGzDecoder, write::GzEncoder};
use prost::Message;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::Path,
};

#[derive(Clone, PartialEq, Message)]
struct Example {
    #[prost(message, optional, tag = "1")]
    features: Option<Features>,
}
#[derive(Clone, PartialEq, Message)]
struct Features {
    #[prost(map = "string, message", tag = "1")]
    feature: HashMap<String, Feature>,
}
#[derive(Clone, PartialEq, Message)]
struct Feature {
    #[prost(message, optional, tag = "1")]
    bytes: Option<BytesList>,
    #[prost(message, optional, tag = "2")]
    floats: Option<FloatList>,
    #[prost(message, optional, tag = "3")]
    ints: Option<IntList>,
}
#[derive(Clone, PartialEq, Message)]
struct BytesList {
    #[prost(bytes = "vec", repeated, tag = "1")]
    value: Vec<Vec<u8>>,
}
#[derive(Clone, PartialEq, Message)]
struct FloatList {
    #[prost(float, repeated, packed = "true", tag = "1")]
    value: Vec<f32>,
}
#[derive(Clone, PartialEq, Message)]
struct IntList {
    #[prost(int64, repeated, packed = "true", tag = "1")]
    value: Vec<i64>,
}

pub fn reader(path: &Path) -> Result<Box<dyn Read>> {
    let file = File::open(path)?;
    if path.extension().is_some_and(|x| x == "gz") {
        Ok(Box::new(MultiGzDecoder::new(file)))
    } else {
        Ok(Box::new(file))
    }
}
pub enum OutputWriter {
    Plain(File),
    Gzip(GzEncoder<File>),
}
impl Write for OutputWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        match self {
            Self::Plain(w) => w.write(bytes),
            Self::Gzip(w) => w.write(bytes),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Self::Plain(w) => w.flush(),
            Self::Gzip(w) => w.flush(),
        }
    }
}
impl OutputWriter {
    pub fn finish(self) -> Result<()> {
        let file = match self {
            Self::Plain(f) => f,
            Self::Gzip(w) => w.finish()?,
        };
        file.sync_all()?;
        Ok(())
    }
}
pub fn writer(path: &Path) -> Result<OutputWriter> {
    let file = File::create(path)?;
    if path.extension().is_some_and(|x| x == "gz") {
        Ok(OutputWriter::Gzip(GzEncoder::new(
            file,
            Compression::fast(),
        )))
    } else {
        Ok(OutputWriter::Plain(file))
    }
}
fn masked_crc(bytes: &[u8]) -> u32 {
    crc32c::crc32c(bytes)
        .rotate_right(15)
        .wrapping_add(0xa282ead8)
}
fn bytes(v: Vec<u8>) -> Feature {
    Feature {
        bytes: Some(BytesList { value: vec![v] }),
        floats: None,
        ints: None,
    }
}
fn ints(v: Vec<i64>) -> Feature {
    Feature {
        bytes: None,
        floats: None,
        ints: Some(IntList { value: v }),
    }
}

fn example(w: &Window, c: &ModelConfig) -> Example {
    let mut f = HashMap::new();
    f.insert(
        "subreads/encoded".into(),
        bytes(w.rows.iter().flat_map(|x| x.to_le_bytes()).collect()),
    );
    f.insert(
        "subreads/shape".into(),
        ints(vec![c.rows() as i64, w.length as i64, 1]),
    );
    f.insert("subreads/num_passes".into(), ints(vec![w.passes as i64]));
    f.insert("name".into(), bytes(w.name.as_bytes().to_vec()));
    f.insert("window_pos".into(), ints(vec![w.window_pos as i64]));
    f.insert(
        "ccs_base_quality_scores".into(),
        ints(w.qualities.iter().map(|&x| x as i64).collect()),
    );
    if let Some(label) = &w.label {
        f.insert(
            "label/encoded".into(),
            bytes(
                label
                    .iter()
                    .flat_map(|&x| (x as f32).to_le_bytes())
                    .collect(),
            ),
        );
        f.insert("label/shape".into(), ints(vec![label.len() as i64]));
    }
    Example {
        features: Some(Features { feature: f }),
    }
}

pub struct DatasetWriter {
    writer: BufWriter<OutputWriter>,
    tfrecord: bool,
    config: ModelConfig,
}

pub fn is_tfrecord(path: &Path) -> Result<bool> {
    let name = if path.extension().is_some_and(|e| e == "gz") {
        Path::new(
            path.file_stem()
                .ok_or_else(|| anyhow::anyhow!("missing dataset filename"))?,
        )
    } else {
        path
    };
    match name.extension().and_then(|e| e.to_str()) {
        Some("tfrecord" | "tfrecords") => Ok(true),
        Some("jsonl") => Ok(false),
        _ => anyhow::bail!(
            "dataset path must end in .tfrecord, .tfrecords, or .jsonl (optionally .gz): {}",
            path.display()
        ),
    }
}

impl DatasetWriter {
    pub fn new(path: &Path, c: &ModelConfig) -> Result<Self> {
        let tfrecord = is_tfrecord(path)?;
        Ok(Self {
            writer: BufWriter::new(writer(path)?),
            tfrecord,
            config: c.clone(),
        })
    }
    pub fn push(&mut self, w: &Window) -> Result<()> {
        if self.tfrecord {
            let bytes = example(w, &self.config).encode_to_vec();
            let len = (bytes.len() as u64).to_le_bytes();
            self.writer.write_all(&len)?;
            self.writer.write_all(&masked_crc(&len).to_le_bytes())?;
            self.writer.write_all(&bytes)?;
            self.writer.write_all(&masked_crc(&bytes).to_le_bytes())?;
        } else {
            serde_json::to_writer(&mut self.writer, w)?;
            self.writer.write_all(b"\n")?;
        }
        Ok(())
    }
    pub fn finish(mut self) -> Result<()> {
        self.writer.flush()?;
        self.writer
            .into_inner()
            .map_err(|e| e.into_error())?
            .finish()?;
        Ok(())
    }
}

pub struct DatasetReader {
    reader: BufReader<Box<dyn Read>>,
    tfrecord: bool,
    config: ModelConfig,
}
impl DatasetReader {
    pub fn new(path: &Path, c: &ModelConfig) -> Result<Self> {
        let tfrecord = is_tfrecord(path)?;
        Ok(Self {
            reader: BufReader::new(reader(path)?),
            tfrecord,
            config: c.clone(),
        })
    }
    pub fn next_window(&mut self) -> Result<Option<Window>> {
        if !self.tfrecord {
            let mut line = String::new();
            if self.reader.read_line(&mut line)? == 0 {
                return Ok(None);
            }
            let w: Window = serde_json::from_str(&line)?;
            self.validate(&w)?;
            return Ok(Some(w));
        }
        let mut len = [0u8; 8];
        if self.reader.read(&mut len[..1])? == 0 {
            return Ok(None);
        }
        self.reader.read_exact(&mut len[1..])?;
        let mut crc = [0u8; 4];
        self.reader.read_exact(&mut crc)?;
        ensure!(
            u32::from_le_bytes(crc) == masked_crc(&len),
            "TFRecord length checksum mismatch"
        );
        let n = u64::from_le_bytes(len);
        ensure!(
            n <= 128 * 1024 * 1024,
            "TFRecord exceeds 128 MiB safety limit"
        );
        let mut data = vec![0; n as usize];
        self.reader.read_exact(&mut data)?;
        self.reader.read_exact(&mut crc)?;
        ensure!(
            u32::from_le_bytes(crc) == masked_crc(&data),
            "TFRecord payload checksum mismatch"
        );
        let f = Example::decode(data.as_slice())?
            .features
            .ok_or_else(|| anyhow::anyhow!("missing tf.Example features"))?
            .feature;
        let get_bytes = |key: &str| -> Result<&[u8]> {
            Ok(f.get(key)
                .and_then(|x| x.bytes.as_ref())
                .and_then(|x| x.value.first())
                .ok_or_else(|| anyhow::anyhow!("missing {key}"))?
                .as_slice())
        };
        let get_ints = |key: &str| -> Result<&[i64]> {
            Ok(&f
                .get(key)
                .and_then(|x| x.ints.as_ref())
                .ok_or_else(|| anyhow::anyhow!("missing {key}"))?
                .value)
        };
        let shape = get_ints("subreads/shape")?;
        ensure!(
            shape == [self.config.rows() as i64, self.config.max_length as i64, 1],
            "TFRecord feature shape mismatch: {shape:?}"
        );
        let raw = get_bytes("subreads/encoded")?;
        ensure!(
            raw.len() == self.config.rows() * self.config.max_length * 4,
            "feature bytes length mismatch"
        );
        let rows = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect::<Vec<_>>();
        let label = if f.contains_key("label/encoded") {
            let raw = get_bytes("label/encoded")?;
            ensure!(
                raw.len() == self.config.max_length * 4,
                "label bytes length mismatch"
            );
            let mut label = Vec::new();
            for v in raw.as_chunks::<4>().0.iter() {
                let v = f32::from_le_bytes(*v);
                ensure!(
                    (0.0..=4.0).contains(&v) && v.fract() == 0.0,
                    "invalid label float"
                );
                label.push(v as u8);
            }
            Some(label)
        } else {
            None
        };
        let length = self.config.max_length;
        let ccs = rows
            [4 * self.config.max_passes * length..(4 * self.config.max_passes + 1) * length]
            .iter()
            .map(|&x| x as u8)
            .collect();
        let qualities = get_ints("ccs_base_quality_scores")?
            .iter()
            .map(|&x| x as i16)
            .collect();
        let passes = get_ints("subreads/num_passes")?;
        let pos = get_ints("window_pos")?;
        ensure!(
            passes.len() == 1 && pos.len() == 1 && passes[0] >= 0 && pos[0] >= 0,
            "invalid window metadata"
        );
        let w = Window {
            name: String::from_utf8(get_bytes("name")?.to_vec())?,
            window_pos: pos[0] as usize,
            length,
            passes: passes[0] as usize,
            rows,
            label,
            ccs,
            qualities,
            overflow: false,
        };
        self.validate(&w)?;
        Ok(Some(w))
    }
    fn validate(&self, w: &Window) -> Result<()> {
        ensure!(
            w.rows.len() == self.config.rows() * w.length
                && w.ccs.len() == w.length
                && w.qualities.len() == w.length
                && w.rows.iter().all(|x| x.is_finite()),
            "invalid dataset window"
        );
        ensure!(
            w.label
                .as_ref()
                .is_none_or(|l| l.len() == self.config.max_length && l.iter().all(|&x| x < 5)),
            "invalid label"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tfrecord_roundtrip_and_crc() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("x.tfrecord.gz");
        let c = ModelConfig {
            max_length: 2,
            max_passes: 1,
            ..Default::default()
        };
        let w = Window {
            name: "m/1/ccs".into(),
            window_pos: 0,
            length: 2,
            passes: 1,
            rows: vec![0.0; c.rows() * 2],
            label: Some(vec![1, 2]),
            ccs: vec![0; 2],
            qualities: vec![30, -1],
            overflow: false,
        };
        let mut writer = DatasetWriter::new(&p, &c).unwrap();
        writer.push(&w).unwrap();
        writer.finish().unwrap();
        let mut r = DatasetReader::new(&p, &c).unwrap();
        let got = r.next_window().unwrap().unwrap();
        assert_eq!(got.label, w.label);
        assert_eq!(got.qualities, w.qualities);
        assert!(r.next_window().unwrap().is_none());
        let folder = d.path().join("tfrecord-directory");
        std::fs::create_dir(&folder).unwrap();
        for name in ["x.jsonl", "x.jsonl.gz", "x.tfrecords", "x.tfrecord"] {
            let path = folder.join(name);
            let mut writer = DatasetWriter::new(&path, &c).unwrap();
            writer.push(&w).unwrap();
            writer.finish().unwrap();
            let mut reader = DatasetReader::new(&path, &c).unwrap();
            let got = reader.next_window().unwrap().unwrap();
            assert_eq!(got.label, w.label);
            assert_eq!(got.qualities, w.qualities);
            assert!(reader.next_window().unwrap().is_none());
        }
    }

    #[test]
    fn dataset_suffix_and_nontruncation() {
        for name in ["x.tfrecord", "x.tfrecord.gz", "x.tfrecords.gz"] {
            assert!(is_tfrecord(Path::new(name)).unwrap());
        }
        assert!(!is_tfrecord(Path::new("tfrecord-folder/x.jsonl.gz")).unwrap());
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("unsupported.data");
        std::fs::write(&path, b"keep").unwrap();
        assert!(DatasetWriter::new(&path, &ModelConfig::default()).is_err());
        assert_eq!(std::fs::read(path).unwrap(), b"keep");
    }
}
