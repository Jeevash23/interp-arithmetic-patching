# Independent reconstruction and the device-norm discrepancy

The final independent audit reconstructs all 360 replacement vectors exactly from recaptured donor/recipient states and frozen bases, checks all 144 reused outputs, all 756 fitting-vector hashes, and all 36 self generations. The learned artifact hash is unchanged.

The first audit stopped when it compared a CPU-computed vector norm against a GPU-recorded norm at absolute tolerance 1e-5. Both computations subtract in bfloat16 and sum squared entries in float32, but their reductions round differently. The largest observed CPU/GPU difference was 0.00002288818359375. The corrected audit explicitly repeats the original GPU arithmetic on the independently recaptured vectors: its maximum discrepancy from every recorded norm is exactly zero. Exact vector equality, source hashes, write locality and all scientific outcomes remain unchanged.

The initial source is `audit-before-device-norm-check.py`, its failed output is `audit.log`, and the completed audit is `audit_subspace.py` with `audit-device-norm.log` and `results/completed-audit.json`. This correction does not relax the experiment's competence or causal-progression gates.
