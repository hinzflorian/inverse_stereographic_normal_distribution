# Shifted Inverse stereographic normal distribution

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/hinzflorian/isnd/blob/main/LICENSE)

## Installation

The package dependencies are stored in environments/isnd_code.yml. Create an environment as follows

```bash
conda env create -f environments/isnd_code.yml
```

## Dataset generation
Further dependencies need to be installed to generate the datasets.

Install bgflow

```bash
git clone https://github.com/noegroup/bgflow.git
cd bgflow
python setup.py install
```

Install bgmol

```bash
git clone https://github.com/noegroup/bgmol.git
cd bgmol
python setup.py install
```

The data sets for alanine tetrapeptide can be generated executing 

```bash
python -m isnd.data_generation.generate_data.py
```

## Training

First activate the conda environment using

```bash
conda activatae isnd_code
```
To fit e.g. the alanine tetrapeptide data, run

```bash
python -m isnd.examples.fit_peptide
```

## Evaluation

```bash
python -m isnd.utils.estimate_divergence
```
