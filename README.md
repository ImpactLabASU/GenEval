# GenEval: Multi-Modal Single-Source Domain Generalization for Medical Imaging

Official PyTorch implementation of **GenEval** (WACV 2026).

**Paper:** [Link to paper when published]  
**Authors:**  Ayan Banerjee, Kuntal Prakash Thakur Sandeep Gupta  
**Affiliation:** IMPACT Lab, Arizona State University

## Overview
GenEval combines multi-modal learning with single-source domain generalization for diabetic retinopathy (DR). It fine-tunes MedGemma-4B with LoRA adapters to classify DR severity grades across multiple retinal datasets.

## Installation
```bash
git clone https://github.com/ImpactLabASU/GenEval.git
cd GenEval
pip install -r requirements.txt
```

## Datasets
Download datasets:
- **APTOS:** [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection)
- **EyePACS:** [Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection)
- **Messidor-1:** [Link]
- **Messidor-2:** [Link]

See `docs/DATASETS.md` for details.

## Training
```bash
python diabetic_retinopathy/train.py \
  --dataset aptos \
  --data_path /path/to/aptos/images \
  --csv_path /path/to/aptos/labels.csv \
  --output_dir ./checkpoints/aptos
```

Multi-domain (MDG-style) example:
```bash
python diabetic_retinopathy/train.py \
  --datasets messidor2 eyepacs \
  --data_paths /data/messidor2/images /data/eyepacs/images \
  --csv_paths /data/messidor2/labels.csv /data/eyepacs/trainLabels.csv \
  --output_dir ./checkpoints/mdg_mix
```

## Evaluation
```bash
python diabetic_retinopathy/evaluate.py \
  --model_path ./checkpoints/aptos \
  --test_dataset eyepacs \
  --test_data_path /path/to/eyepacs/images \
  --test_csv_path /path/to/eyepacs/labels.csv
```

## Pretrained Models
LoRA adapters available on Dropbox:
```bash
https://www.dropbox.com/scl/fo/e51mect061togeqdwq2wg/ALCG9_9IPHLExKlpghC_6ks?rlkey=4xrh2df0y3xnv5yph1pbc91am&st=jhpgto0w&dl=0
```

## Citation
```bibtex
@inproceedings{banerjee2026_humanknowledge,
  title={Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization},
  author={Banerjee, Ayan and Thakur, Kuntal and Gupta, Sandeep},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2026},
  url={https://openaccess.thecvf.com/content/WACV2026/html/Banerjee_Human_Knowledge_Integrated_Multi-modal_Learning_for_Single_Source_Domain_Generalization_WACV_2026_paper.html}
}
```

## License
MIT License
