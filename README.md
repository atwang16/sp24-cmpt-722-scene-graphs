# Evaluating Text-to-Scene-Graph Methods for 3D Scene Generation

CMPT 722, Spring 2024

Authors: Austin Wang, Colin Li, ZeMing Gong

## Setup

### Environment

```bash
pip install -e .
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
