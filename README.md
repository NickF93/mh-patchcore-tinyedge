# mh-patchcore-tinyedge

`mh-patchcore-tinyedge` is a small runtime for PatchCore-style anomaly scoring on
edge devices. It loads a compact student checkpoint, builds a memory bank from
normal images, and scores new images with exact squared-L2 nearest-neighbor
distances.

The runtime path is intentionally narrow:

```text
image -> student features -> nearest distance to student memory bank -> score
```

Training is not part of this package. The runtime only needs the checkpoint,
normal reference images for the memory bank, and the images you want to score.

## Get Teacher Artifacts

Student training starts from teacher artifacts produced by the public
`MH-PatchCore` project. From an `MH-PatchCore` checkout, place MVTec AD under
`mvtec_datasets/` and run:

```bash
python run_mhpc.py --config configs/mvtec/teacher/mvtec_streaming_mh_patchcore_teacher_artifacts.yaml
```

The run writes the teacher payloads under:

```text
results/mvtec/mvtec_streaming_mh_patchcore_teacher_artifacts/<timestamp>/artifacts/
```

Use that `artifacts` directory as the teacher-artifact root for student
training. The export contains the fitted teacher checkpoint, the teacher memory
bank, and frozen replay features grouped by dataset and sample group.

## Install

From a local checkout:

```bash
python -m pip install -e .
```

For tests:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

## Build A Memory Bank

Build the memory bank from normal reference images:

```bash
mh-tinyedge bank build \
  --checkpoint path/to/last.pt \
  --images path/to/normal-images \
  --output runtime/bank.npz
```

The image directory is scanned recursively. Supported formats are PNG, JPEG,
BMP, TIFF, and WebP.

## Score Images

Score a single image or a folder:

```bash
mh-tinyedge score \
  --checkpoint path/to/last.pt \
  --bank runtime/bank.npz \
  --input path/to/images \
  --output runtime/scores.csv \
  --heatmap-dir runtime/heatmaps
```

The score is the maximum patch distance for the image. Higher scores indicate
that at least one patch is farther from the normal memory bank.

`scores.csv` contains:

- `path`: scored image path;
- `score`: anomaly score;
- `heatmap`: optional grayscale heatmap path.

## Inspect Runtime Files

```bash
mh-tinyedge inspect \
  --checkpoint path/to/last.pt \
  --bank runtime/bank.npz
```

This prints the checkpoint feature dimension, memory-bank size, output grid, and
runtime device.

## Device Selection

All commands accept `--device`:

- `auto`: use CUDA when available, otherwise CPU;
- `cpu`;
- `cuda`;
- `cuda:0`, `cuda:1`, and other indexed CUDA devices.

## Runtime Files

The memory bank is stored as a compressed `.npz` file containing:

- feature vectors;
- feature dimension;
- output grid size.

The checkpoint must include the model configuration, preprocessing settings,
feature dimension, output grid, and model weights.
