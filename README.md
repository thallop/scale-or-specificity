This repository contains the code associated with the scientific publication:  
**"Scale or Specificity? Pretraining Foundation Models for Urban Remote Sensing"**

### 📂 Structure
#### Pretraining
Preparing urban-specific datasets and pretraining remote sensing foundation models.
  - `urban_metadata.py` → Compute urban metadata for each sample in SSL4EO-S12.
  - `build_subset.py` → Create sharded dataset subsets from previously generated lists of sample IDs.
  - `dataset.py` → Dataset loading and preprocessing, building upon [DLR-MF-DAS/SSL4EO-S12-v1.1](https://github.com/DLR-MF-DAS/SSL4EO-S12-v1.1).
  - `pretraining.py` → Self-supervised pretraining based on [antofuller/CROMA](https://github.com/antofuller/CROMA).
#### Evaluation
Evaluation of pretrained models on our urban-benchmark:
  - `EuroSAT.ipynb` → Urban LULC classification on EuroSAT.
  - `LCZ42.ipynb` → Urban local climate zone classification on So2Sat LCZ42.
  - `IDEABench.ipynb` → Urban deprivation segmentation on IDEABench.
  - `50Cities.ipynb` → Urban LULC segmentation on 50 Cities.

### 📄 Citation
If you use this code or build upon this work, please cite:

```bibtex
@inproceedings{hallopeauScaleSpecificityPretraining2027,
  title     = {Scale or Specificity? Pretraining Foundation Models for Urban Remote Sensing},
  author    = {Hallopeau, Thomas and Guérin, Joris and Demagistri, Laurent and Mougenot, Isabelle},
  year      = {2027},
  booktitle = {To appear}
}
