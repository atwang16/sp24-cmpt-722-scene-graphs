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

1. Download the [Visual Genome dataset](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html):
```bash
mkdir -p .data/datasets/vg
cd .data/datasets/vg
wget https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/scene_graphs.json.zip
wget https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip

unzip scene_graphs.json.zip
unzip region_descriptions.json.zip
```

2. Download the SG-FRONT scene graph dataset from CommonScenes:
```bash
mkdir -p .data/datasets/3dfront
cd .data/datasets/3dfront
wget https://www.campar.in.tum.de/public_datasets/2023_commonscenes_zhai/SG_FRONT.zip
unzip SG_FRONT.zip
```

3. Download the scene description dataset from InstructScene:
```bash
mkdir -p .data/datasets/3dfront
cd .data/datasets/3dfront
wget https://huggingface.co/datasets/chenguolin/InstructScene_dataset/resolve/main/InstructScene.zip?download=true
```

## Usage

Evaluation:
```bash
evaluate-sg data=3dfront_bedroom model=llm
```

`data`: `3dfront_bedroom`, `3dfront_diningroom`, `3dfront_livingroom`, `vg`

`model`: `vanila_parser`, `llm`, `instructscene_bedroom`, `instructscene_diningroom`, `instructscene_livingroom`
