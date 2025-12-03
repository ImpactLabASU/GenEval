# Diabetic Retinopathy Datasets

## APTOS 2019 Blindness Detection
- **Source:** Kaggle Competition  
- **URL:** https://www.kaggle.com/c/aptos2019-blindness-detection  
- **Description:** Retinal photographs labeled for diabetic retinopathy severity levels (0-4).  
- **Images:** ~3,662 training images, ~1,928 test images.  
- **Format:** JPEG with CSV labels.  
- **Classes:** 5 severity levels (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR).

## Messidor
- **Source:** Digital Retinal Images for Vessel Extraction (DRIVE) project.  
- **URL:** http://www.adcis.net/en/third-party/messidor/  
- **Description:** 1,200 eye fundus images captured using a Topcon TRC NW6 retinograph.  
- **Format:** TIFF images with three native resolutions (1440×960, 2240×1488, 2304×1536).  
- **Classes:** 4 diabetic retinopathy grades (0-3).

## Messidor-2
- **Source:** MESSIDOR-2 MA database.  
- **URL:** http://www.adcis.net/en/third-party/messidor2/  
- **Description:** Extension of MESSIDOR with additional annotations for microaneurysms and diabetic retinopathy grading.  
- **Images:** 1,748 fundus images.  
- **Format:** TIFF images.  
- **Classes:** 5 diabetic retinopathy grades (0-4) aligned with international standards.  
- **Notes:** Includes microaneurysm detection annotations.

## EyePACS
- **Source:** Eye Picture Archive Communication System (EyePACS).  
- **URL:** https://www.kaggle.com/c/diabetic-retinopathy-detection  
- **Description:** Large-scale telemedicine dataset for diabetic retinopathy screening.  
- **Images:** ~35,126 training images, ~53,576 test images.  
- **Format:** JPEG + CSV labels.  
- **Classes:** 5 severity levels (0-4).  
- **Notes:** Standard benchmark for cross-domain generalization studies.

## Dataset Statistics Summary
| Dataset    | Images   | Classes   | Resolution        | Format |
|------------|----------|-----------|-------------------|--------|
| APTOS      | ~5,590   | 5 (0-4)   | Variable          | JPEG   |
| Messidor   | 1,200    | 4 (0-3)   | 1440×960+         | TIFF   |
| Messidor-2 | 1,748    | 5 (0-4)   | Variable          | TIFF   |
| EyePACS    | ~88,702  | 5 (0-4)   | Variable          | JPEG   |

## Usage Notes
- All retinal datasets follow comparable grading scales enabling transfer evaluation.  
- EyePACS is the largest corpus and best suited for large-scale LoRA adaptation.  
- Messidor variants contain high-quality clinical images helpful for validation.  
- APTOS offers competition-grade evaluation benchmarks.  
- Cross-dataset evaluation underpins the single-source domain generalization experiments.

## Citations
- APTOS 2019 Blindness Detection (Kaggle).  
- Decencière et al., "Feedback on a publicly distributed database..." (MESSIDOR).  
- Abramoff et al., "Improved automated detection of diabetic retinopathy..." (MESSIDOR-2).  
- Kaggle Diabetic Retinopathy Detection (EyePACS).  
