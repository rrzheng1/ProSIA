# ProSIA

ProSIA is a multimodal model for unified lysine PTM site prediction using
ESM2, ProtT5, and structural representations with progressive gated fusion.

## Architecture

- ESM2 window: two multi-scale CNN blocks, producing a 600-dimensional site representation.
- ProtT5 window: linear projection, BiGRU, and mean pooling.
- Structure window: linear projection, BiGRU, and mean pooling.
- Gate 1: feature-wise fusion of ESM2 and ProtT5.
- Gate 2: feature-wise fusion of the PLM representation and structure.
- Output: one task-specific classification head for each PTM type.

The ESM2 branch operates on the 31-residue site window through two
multi-scale CNN blocks.

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```

Expected project-local directories:

```text
data/
esm2_embedding/
prott5_embedding/
structure_embedding/
```

## Training

```bash
python run_mt_esm_prott5_gated_structure.py \
  --experiment-name mt_esm_prott5_global_split \
  --batch-size 16 \
  --device cuda
```

For architecture details and the gated-fusion equations, see
[`README_STRUCTURE.md`](README_STRUCTURE.md).
