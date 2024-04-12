# Evaluating Text-to-Scene-Graph Methods for 3D Scene Generation

CMPT 722, Spring 2024

Authors: Austin Wang, Colin Li, ZeMing Gong

## Setup

### Environment

```bash
pip install -e .
python -m spacy download en
```

### Data

1. Download the SG-FRONT scene graph dataset from CommonScenes:
```bash
mkdir -p .data/datasets/3dfront
cd .data/datasets/3dfront
wget https://www.campar.in.tum.de/public_datasets/2023_commonscenes_zhai/SG_FRONT.zip
unzip SG_FRONT.zip
```

2. The pretrained diffusion model for InstructScene can be downloaded [here](https://vault.sfu.ca/index.php/s/QLc972akWzlw3K7).

## Usage

**Generate** a scene graph from a single text prompt:
```bash
generate data=3dfront_bedroom model=llm
```

**Evaluation**: This script generates scene graphs for a dataset of scene descriptions, computes metrics against ground truth scene graphs, and generates scene graph visualizations. The output format includes a JSON compatible with CommonScenes as well as a JSON of all generated results. See `eval_config.yaml` for more configuration options.

```bash
evaluate-sg data=3dfront_bedroom model=llm
```

### Reference

`data`: `3dfront_bedroom`, `3dfront_diningroom`, `3dfront_livingroom`

`model`: `vanila_parser`, `llm`, `instructscene_bedroom`, `instructscene_diningroom`, `instructscene_livingroom`
