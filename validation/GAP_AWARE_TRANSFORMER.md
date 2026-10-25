# What DeepConsensus Means by a Gap-Aware Transformer

## Core Idea

DeepConsensus is a transformer encoder for correcting a draft PacBio circular
consensus sequence from repeated observations of the same molecule. Its
important adaptation is the combination of a gapped alignment representation,
base-or-gap predictions, and an alignment-aware differentiable training loss.
"Gap-aware" does not mean a new Q/K/V attention formula or an adaptive attention
band. The [original paper](https://www.nature.com/articles/s41587-022-01435-7)
identifies the alignment-based loss as central to training the encoder.

There are three separate components to keep straight: constructing the input
alignment, the transformer's self-attention, and aligning predictions to truth
inside the training loss. They solve different problems.

## 1. Why Sequence Gaps Matter

PacBio observes the same circularized DNA molecule multiple times. Each pass
can contain substitutions, insertions, and deletions. Consequently, raw index
10 in one pass need not represent the same underlying base as index 10 in
another. An alignment inserts empty slots so corresponding evidence can be
placed into shared columns.

Here is a schematic example, not data from the benchmark:

```text
Draft CCS:  A C - G T
Pass 1:     A C A G T
Pass 2:     A C A G T
Pass 3:     A C - G T
```

The third column exists because some passes contain an insertion relative to
the draft. A gap means no base occupies that particular alignment slot; it is
not an unknown base such as `N`. The network can use the evidence to emit `A`
there, changing `ACGT` to `ACAGT`. Conversely, it can emit a gap where the draft
has an extra base and remove that base from the corrected sequence.

The [preprocessing implementation](https://github.com/google/deepconsensus/blob/r1.2/deepconsensus/preprocess/pre_lib.py)
receives subreads already aligned to their CCS molecule. Shared insertion
columns are constructed before model inference. The transformer does not start
from arbitrary unaligned raw sequences, nor does its training-loss DP create
the input subread alignment.

## 2. What the Encoder Sees

For each alignment column, the model receives base calls and sequencing
evidence from multiple passes, including pulse width, interpulse duration,
strand, and signal-to-noise information, together with the draft CCS base.
The later v1.2 model also uses draft CCS quality. Learned feature embeddings
are combined into one vector per alignment column. Positional information is
added, and encoder blocks exchange context along the sequence.

The [released learned-values network](https://github.com/google/deepconsensus/blob/r1.2/deepconsensus/models/networks.py)
mixes the pass/feature rows through embedding concatenation and a linear
condenser. Its self-attention acts along alignment positions, not as independent
two-dimensional attention over both the pass axis and the position axis.

Our port uses the actual published v1.2 student checkpoint: 100 columns, up to
20 feature passes, 86 raw feature rows, a 568-channel concatenated embedding,
and a condenser to 280 channels. There are five encoder layers with two heads
and a ReLU feed-forward width of 2048. Those are release/checkpoint details,
not a claim that the original paper's earlier model had this exact shape.

The input base embedding for gap ID zero is explicitly zeroed. That masking
does not remove the alignment column or force its predicted output to be a
gap. Other passes and neighboring columns can provide evidence for a base.

## 3. What the Network Predicts

The classifier emits five probabilities per alignment column:

```text
P(gap), P(A), P(T), P(C), P(G)
```

This matches the release's vocabulary order. At inference, the highest
probability class is selected at each position and predicted gaps are removed.
Thus a fixed-size tensor can yield different-length DNA strings.

This is not an autoregressive decoder that can invent arbitrarily many extra
positions. It can fill insertion slots present in the alignment representation
and remove unwanted columns; the available output slots still limit length.
The published implementation also has policies for padding, overflow, window
skipping, quality calibration, and stitching.

## 4. Why Column-Wise Cross-Entropy Is Inadequate

Repeats make gap placement ambiguous. Consider these two aligned outputs:

```text
Output 1:  A A A - C
Output 2:  A A - A C
```

Both spell exactly `AAAC` after removing gaps. A loss that requires the same
target class at each fixed column can penalize columns three and four even
though the biological sequence is unchanged. It effectively treats one
arbitrary placement of the gap as the only correct answer.

DeepConsensus instead compares the predicted sequence distributions to the
ungapped truth through a monotonic alignment. This allows a base prediction
to correspond to an appropriate truth base even when their original column
indices differ. It addresses correspondence ambiguity, not just the total
number of mistakes in a long read.

## 5. The Differentiable Alignment Loss

Let truth bases be `t_1 ... t_m`, and let `p_j` be the predicted distribution
at alignment column `j`, with `n` predicted columns. In schematic form, the
loss dynamic program is:

```text
D(i,j) = softmin_tau(
    D(i-1,j-1) - log p_j(t_i),
    D(i,  j-1) - log p_j(gap),
    D(i-1,j  ) + deletion_cost
)
```

The transitions mean:

- Consume a truth base and an output column, paying the negative log
  probability of that truth base at that column.
- Consume only an output column, paying its gap cost. An unused alignment slot
  should therefore have a high predicted gap probability.
- Consume only a truth base, paying the fixed deletion penalty for a truth
  character with no corresponding output slot.

Boundary conditions account for initial skipped bases/columns. The final
cell compares the complete truth with the complete output grid. The release
also clips probabilities for numerical stability and handles variable truth
lengths and padded tensors. See the [actual loss implementation](https://github.com/google/deepconsensus/blob/r1.2/deepconsensus/models/losses_and_metrics.py).

For the entropy-smoothed minimum used with positive regularization:

```text
softmin_tau(c_1,...,c_k) = -tau * log(sum(exp(-c_k / tau)))
```

The code computes this stably. As smoothing tends toward zero, it approaches
the hard minimum. With positive smoothing, multiple plausible alignments
contribute to the objective and its gradient. Backpropagation can then tell
the network which base/gap probabilities to increase or decrease without
requiring one fixed ground-truth gap placement. This follows the general
[differentiable dynamic programming framework](https://proceedings.mlr.press/v80/mensch18a.html)
that the DeepConsensus paper cites.

This is not simply an ordinary integer edit-distance score added after
prediction. The transition costs depend on network probabilities, and the
soft recurrence must supply gradients. A conventional hard-score DADA2,
Needleman-Wunsch, or WFA aligner is not a drop-in replacement for this objective.
The familiar alignment geometry does not make their scoring/backward behavior
equivalent.

Equivalent gapped spellings need not have exactly identical numerical soft
losses: per-position probabilities, smoothing, clipping, boundaries, and any
finite band still matter. The useful property is flexible sequence
correspondence instead of compulsory column-by-column target matching.

## 6. Inference Does Not Run the Training Alignment DP

The usual correction path is:

```text
Aligned BAMs -> shared gap columns -> feature windows -> encoder
            -> per-column base/gap calls -> remove gaps
            -> calibrate qualities -> stitch windows -> FASTQ
```

Training adds the differentiable alignment comparison to reference labels and
backpropagation. Evaluation may also compute alignment loss and edit metrics.
Production neural inference does not search the loss DP for a best corrected
sequence. It already has a trained encoder and decodes its output classes.

This distinction explains why accelerating the loss chiefly helps training,
whereas accelerating feature preparation, encoder matrices, or ONNX execution
can help inference.

## 7. Three Different Meanings of a Band

- An input alignment determines which subread observations occupy shared
  columns. Upstream alignment algorithms may themselves use banded DP.
- Local attention masks which neighboring model positions a head may attend
  to. The published v1.2 checkpoint uses radius 12; multiple encoder layers
  propagate context beyond one layer's immediate radius.
- A banded alignment loss restricts which prediction-to-truth DP paths are
  considered. The published v1.2 checkpoint parameters specify an unbanded
  loss; later work introduces endpoint-aware convex-hull band geometry.

Neither local attention nor convex-hull loss banding is the definition of
"gap-aware." A wider endpoint region is about plausible alignment geometry
between different-length axes; it does not establish that error rate must
increase toward the end. More bases can accumulate more total errors even
under a constant per-base error rate.

## Relation to the Rust Optimization

The optimization should preserve the evidence layout, output vocabulary,
attention connectivity, trained weights, and loss/backward contract. Folding
embedding scaling or changing matrix scheduling can reduce implementation
overhead without changing the intended model. Narrowing the attention radius,
pruning loss paths, removing passes, quantizing weights, or skipping more
windows changes behavior and requires separate accuracy validation.

The 6,400-window comparison uses real labels and the dense original loss,
so a speed result is not obtained by silently switching to an easier objective.
CPU results cannot predict CUDA/TensorRT performance without measurements.
