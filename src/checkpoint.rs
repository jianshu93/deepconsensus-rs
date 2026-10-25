use crate::{config::ModelConfig, model::DeepConsensus};
use anyhow::{Context, Result, ensure};
use burn::{
    module::{Module, Param},
    prelude::*,
};
use prost::Message;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

#[derive(Clone, PartialEq, Message)]
struct BundleEntry {
    #[prost(int32, tag = "1")]
    dtype: i32,
    #[prost(message, optional, tag = "2")]
    shape: Option<Shape>,
    #[prost(int32, tag = "3")]
    shard: i32,
    #[prost(int64, tag = "4")]
    offset: i64,
    #[prost(int64, tag = "5")]
    size: i64,
    #[prost(fixed32, tag = "6")]
    crc: u32,
}
#[derive(Clone, PartialEq, Message)]
struct Shape {
    #[prost(message, repeated, tag = "2")]
    dims: Vec<Dim>,
}
#[derive(Clone, PartialEq, Message)]
struct Dim {
    #[prost(int64, tag = "1")]
    size: i64,
}
#[derive(Clone, PartialEq, Message)]
struct BundleHeader {
    #[prost(int32, tag = "1")]
    shards: i32,
    #[prost(int32, tag = "2")]
    endianness: i32,
}

fn varint(data: &[u8], pos: &mut usize) -> Result<u64> {
    let mut v = 0;
    for s in (0..70).step_by(7) {
        ensure!(*pos < data.len() && s < 64, "invalid SSTable varint");
        let x = data[*pos];
        *pos += 1;
        v |= ((x & 127) as u64) << s;
        if x & 128 == 0 {
            return Ok(v);
        }
    }
    anyhow::bail!("varint overflow")
}
fn handle(data: &[u8]) -> Result<(usize, usize)> {
    let mut p = 0;
    let offset = varint(data, &mut p)?;
    let size = varint(data, &mut p)?;
    Ok((offset.try_into()?, size.try_into()?))
}
fn block(table: &[u8], offset: usize, size: usize) -> Result<Vec<u8>> {
    let end = offset
        .checked_add(size)
        .ok_or_else(|| anyhow::anyhow!("SST block overflow"))?;
    ensure!(end + 5 <= table.len(), "SST block outside file");
    let stored = u32::from_le_bytes(table[end + 1..end + 5].try_into()?);
    let crc = crc32c::crc32c(&table[offset..end + 1])
        .rotate_right(15)
        .wrapping_add(0xa282ead8);
    ensure!(stored == crc, "SST block checksum mismatch");
    match table[end] {
        0 => Ok(table[offset..end].to_vec()),
        1 => Ok(snap::raw::Decoder::new().decompress_vec(&table[offset..end])?),
        v => anyhow::bail!("unsupported SST compression {v}"),
    }
}
fn entries(data: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    ensure!(data.len() >= 4, "truncated SST block");
    let restarts = u32::from_le_bytes(data[data.len() - 4..].try_into()?) as usize;
    let limit = data
        .len()
        .checked_sub((restarts + 1) * 4)
        .ok_or_else(|| anyhow::anyhow!("invalid restart table"))?;
    let mut p = 0;
    let mut key = Vec::new();
    let mut entries = Vec::new();
    while p < limit {
        let shared = varint(data, &mut p)? as usize;
        let unshared = varint(data, &mut p)? as usize;
        let size = varint(data, &mut p)? as usize;
        ensure!(
            shared <= key.len() && p + unshared + size <= limit,
            "invalid SST entry"
        );
        key.truncate(shared);
        key.extend_from_slice(&data[p..p + unshared]);
        p += unshared;
        entries.push((key.clone(), data[p..p + size].to_vec()));
        p += size;
    }
    Ok(entries)
}

pub struct TensorflowCheckpoint {
    prefix: String,
    header: BundleHeader,
    entries: BTreeMap<String, BundleEntry>,
}
impl TensorflowCheckpoint {
    pub fn open(prefix: &Path) -> Result<Self> {
        let prefix = prefix.to_string_lossy().into_owned();
        let data = std::fs::read(format!("{prefix}.index"))?;
        ensure!(data.len() >= 48, "truncated TensorFlow checkpoint index");
        let footer = &data[data.len() - 48..];
        ensure!(
            u64::from_le_bytes(footer[40..48].try_into()?) == 0xdb4775248b80fb57,
            "not a TensorFlow SSTable index"
        );
        let mut p = 0;
        varint(footer, &mut p)?;
        varint(footer, &mut p)?;
        let (offset, size) = handle(&footer[p..40])?;
        let index = entries(&block(&data, offset, size)?)?;
        let mut tensors = BTreeMap::new();
        let mut header = None;
        for (_, h) in index {
            let (offset, size) = handle(&h)?;
            for (key, value) in entries(&block(&data, offset, size)?)? {
                if key.is_empty() {
                    header = Some(BundleHeader::decode(value.as_slice())?);
                } else {
                    tensors.insert(
                        String::from_utf8(key)?,
                        BundleEntry::decode(value.as_slice())?,
                    );
                }
            }
        }
        let header = header.ok_or_else(|| anyhow::anyhow!("checkpoint header missing"))?;
        ensure!(
            header.shards > 0 && header.endianness == 0,
            "unsupported checkpoint byte order/sharding"
        );
        Ok(Self {
            prefix,
            header,
            entries: tensors,
        })
    }
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }
    pub fn tensor(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let e = self
            .entries
            .get(name)
            .with_context(|| format!("checkpoint tensor missing: {name}"))?;
        ensure!(
            e.dtype == 1
                && e.offset >= 0
                && e.size >= 0
                && e.shard >= 0
                && e.shard < self.header.shards,
            "not an ordinary F32 tensor: {name}"
        );
        let shape: Vec<usize> = e
            .shape
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing shape"))?
            .dims
            .iter()
            .map(|d| usize::try_from(d.size))
            .collect::<std::result::Result<_, _>>()?;
        let size = shape
            .iter()
            .try_fold(4usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| anyhow::anyhow!("tensor size overflow"))?;
        ensure!(size == e.size as usize, "tensor size/shape mismatch");
        let mut f = File::open(format!(
            "{}.data-{:05}-of-{:05}",
            self.prefix, e.shard, self.header.shards
        ))?;
        ensure!(
            e.offset as u64 + e.size as u64 <= f.metadata()?.len(),
            "tensor outside shard"
        );
        f.seek(SeekFrom::Start(e.offset as u64))?;
        let mut bytes = vec![0; size];
        f.read_exact(&mut bytes)?;
        let crc = crc32c::crc32c(&bytes);
        ensure!(
            e.crc == crc || e.crc == crc.rotate_right(15).wrapping_add(0xa282ead8),
            "tensor checksum mismatch: {name}"
        );
        Ok((
            shape,
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|x| f32::from_le_bytes(*x))
                .collect(),
        ))
    }
}

fn set<const D: usize>(
    param: &mut Param<Tensor<D>>,
    checkpoint: &TensorflowCheckpoint,
    name: &str,
    device: &Device,
) -> Result<()> {
    let (_, data) = checkpoint.tensor(&format!("{name}/.ATTRIBUTES/VARIABLE_VALUE"))?;
    let shape = param.val().dims();
    ensure!(
        data.len() == shape.iter().product::<usize>(),
        "shape mismatch for {name}"
    );
    *param = param
        .clone()
        .map(|_| Tensor::from_data(TensorData::new(data, shape), device));
    Ok(())
}

pub fn import_tensorflow(prefix: &Path, c: &ModelConfig, device: &Device) -> Result<DeepConsensus> {
    let checkpoint = TensorflowCheckpoint::open(prefix)?;
    let mut m = DeepConsensus::new(c, device)?;
    for (param, name) in [
        (&mut m.bases.weight, "bases_embedding_layer"),
        (&mut m.pw.weight, "pw_embedding_layer"),
        (&mut m.ip.weight, "ip_embedding_layer"),
        (&mut m.strand.weight, "strand_embedding_layer"),
        (&mut m.bq.weight, "ccs_base_quality_scores_embedding_layer"),
        (&mut m.sn.weight, "sn_embedding_layer"),
    ] {
        if name == "ccs_base_quality_scores_embedding_layer" && !c.use_ccs_bq {
            continue;
        }
        set(
            param,
            &checkpoint,
            &format!("model/{name}/embeddings"),
            device,
        )?;
    }
    set(
        &mut m.condenser.weight,
        &checkpoint,
        "model/transformer_input_condenser/kernel",
        device,
    )?;
    set(
        &mut m.classifier.weight,
        &checkpoint,
        "model/fc1/kernel",
        device,
    )?;
    set(
        m.classifier.bias.as_mut().unwrap(),
        &checkpoint,
        "model/fc1/bias",
        device,
    )?;
    set(
        &mut m.final_norm.gamma,
        &checkpoint,
        "model/encoder_stack/output_normalization/gamma",
        device,
    )?;
    set(
        m.final_norm.beta.as_mut().unwrap(),
        &checkpoint,
        "model/encoder_stack/output_normalization/beta",
        device,
    )?;
    for (i, b) in m.blocks.iter_mut().enumerate() {
        let prefix = format!("model/encoder_stack/layers/{i}");
        for (param, layer) in [
            (&mut b.query.weight, "query_dense_layer"),
            (&mut b.key.weight, "key_dense_layer"),
            (&mut b.value.weight, "value_dense_layer"),
            (&mut b.output.weight, "output_dense_layer"),
        ] {
            set(
                param,
                &checkpoint,
                &format!("{prefix}/0/layer/{layer}/kernel"),
                device,
            )?;
        }
        set(
            &mut b.expand.weight,
            &checkpoint,
            &format!("{prefix}/1/layer/filter_dense_layer/kernel"),
            device,
        )?;
        set(
            b.expand.bias.as_mut().unwrap(),
            &checkpoint,
            &format!("{prefix}/1/layer/filter_dense_layer/bias"),
            device,
        )?;
        set(
            &mut b.contract.weight,
            &checkpoint,
            &format!("{prefix}/1/layer/output_dense_layer/kernel"),
            device,
        )?;
        set(
            b.contract.bias.as_mut().unwrap(),
            &checkpoint,
            &format!("{prefix}/1/layer/output_dense_layer/bias"),
            device,
        )?;
        if c.rezero {
            set(
                &mut b.attention_alpha,
                &checkpoint,
                &format!("{prefix}/0/alpha"),
                device,
            )?;
            set(
                &mut b.ffn_alpha,
                &checkpoint,
                &format!("{prefix}/1/alpha"),
                device,
            )?;
        } else {
            for (norm, j) in [(&mut b.attention_norm, 0), (&mut b.ffn_norm, 1)] {
                set(
                    &mut norm.gamma,
                    &checkpoint,
                    &format!("{prefix}/{j}/layer_norm/gamma"),
                    device,
                )?;
                set(
                    norm.beta.as_mut().unwrap(),
                    &checkpoint,
                    &format!("{prefix}/{j}/layer_norm/beta"),
                    device,
                )?;
            }
        }
    }
    Ok(m)
}

pub fn save(model: &DeepConsensus, directory: &Path) -> Result<()> {
    std::fs::create_dir_all(directory)?;
    model.config.save(directory.join("params.json"))?;
    let temporary = directory.join("model.tmp.burnpack");
    model.clone().save_file(&temporary)?;
    std::fs::rename(temporary, directory.join("model.burnpack"))?;
    Ok(())
}
pub fn load(directory: &Path, device: &Device) -> Result<DeepConsensus> {
    let c = ModelConfig::load(directory.join("params.json"))?;
    Ok(DeepConsensus::new(&c, device)?.try_load_file(directory.join("model.burnpack"))?)
}
