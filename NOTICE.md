# Attribution

The architecture, preprocessing contracts, calibration behavior, and alignment
objective are reimplemented from Google DeepConsensus under its BSD-3-Clause
license. Original release inspected: r1.2 commit
`979108ddbc0a950ea493071690507ac32c8ee9cb`.

The endpoint-convex-hull band geometry follows the separately released
`banded_alignment` branch. The CPU compact-loss derivative and device wavefront
are implementations of that objective, not DADA2 Needleman-Wunsch kernels.

Original DeepConsensus copyright and redistribution conditions are retained in
LICENSE. Cite the original DeepConsensus paper when using this implementation
in research. This implementation has not been endorsed by Google or PacBio.

The published model and fixture download scripts retrieve artifacts directly
from their upstream locations. This source distribution does not include those
downloads or assert additional licensing rights over them.

Dependencies retain their own licenses. Burn and ORT are MIT/Apache-2.0;
rust-htslib/HTSlib supply native BAM parsing. Consult Cargo metadata for the
complete dependency inventory.
