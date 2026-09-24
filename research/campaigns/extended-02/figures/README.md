# Extended-02 scientific figure source

Status: audit-bound extraction complete; rendering pending in an existing matplotlib environment. Local Python has no matplotlib, so no dependency was installed and no remote render was launched. This package reads only extended-02 evidence and does not alter historical campaigns.

```sh
python3 research/tools/campaign02_figures.py --root . --extract-only
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 research/tools/campaign02_figures.py --render-data research/campaigns/extended-02/figures/plotted-data.json.gz
```

The prepared renderer writes PNG, SVG and PDF for three figures:

- `acquisition`: E03 easy success curves; E05 mixed success and utility curves. Both retain the fixed update-600 endpoint and all six repeated 128-world DEV evaluations. The different curricula/teachers/inputs do not support a controlled forgetting claim.
- `frozen-transfer`: E04 all seven cells and five policies, exact success counts and recorded utility. Every cell has 512 paired worlds; policy repetitions do not expand independent support. Supplied references receive richer raw public JSON than learned encodings, and recorded utility prices compute at zero.
- `reference-budget-repair`: E02 standard/tight budget results under the fixed versus remaining-budget call menu, with original and route-fallback references. These are supplied policy/action-menu repairs, not learning; 4096 archived episodes reuse 512 underlying worlds. The ordinate is explicitly zoomed.

There is one initializer per learned family, not replicated architecture evidence or evolved populations. Teacher-rollout training success and learned closed-loop DEV success are different measurements. No bootstrap or new inferential metric is computed here; E05 and E02 audited paired tables are retained in the data.

`plotted-data.json.gz` deterministically compresses compact sorted JSON with mtime zero. Its 67 unique rows preserve plotted source values; `data-manifest.json` records all 63 audit/raw/summary hashes and compressed/uncompressed table hashes. Each raw binding is verified before extraction. E03 uses its audit's repository-relative source bindings; E04/E05 relative raw bindings are resolved only within their documented result roots; E02 explicitly verifies both archive and summary digests. Auditors' replay limitations remain in the table.

Extraction took 0.031068673997651786 wall seconds. No render was attempted, so no figure is claimed visually verified. Root can render on the existing CPU environment and inspect layouts before publication. No model/solver inference, GPU work or dependency changes occurred. Render cost and final image verification should be recorded after that step.

## Closure render
Rendered PNG/SVG/PDF on the configured GB10 Python environment with CUDA disabled. All three PNG layouts visually inspected by root. CPU launch/accounting receipt: `research/results/campaign-02/closure-figures/occupancy.json`. Earlier pending-render text above records the preparation boundary. No inference or dependency installation.
