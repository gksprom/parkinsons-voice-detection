# Parkinson's Disease Voice Detection

Web application για ανίχνευση της νόσου Πάρκινσον από φωνητικά δείγματα μέσω Machine Learning (Random Forest + SVM ensemble).

Σχολική εργασία. Δεν αποτελεί ιατρικό εργαλείο.

## Επισκόπηση

3-model ensemble εκπαιδευμένο σε 3 διαφορετικά public datasets:

| Dataset | Task | Subjects | Best Model | CV Accuracy |
|---------|------|----------|------------|-------------|
| **UCI Sakar 2019** | Sustained vowel /a/ | 252 (balanced to 128) | RF + SVM | 0.688 |
| **Iyer 2023 (Figshare)** | Sustained vowel /a/ | 81 | RF + SVM | 0.728 |
| **MDVR-KCL Jaeger 2019** | Reading task | 38 | RF + SVM | 0.708 |

Η εφαρμογή έχει 2 στάδια:
1. **Παρατεταμένο φωνήεν "αααα"** → UCI + Iyer ensemble (κάθε ένα: RF + SVM)
2. **Ανάγνωση ελληνικού κειμένου** → MDVR ensemble (RF + SVM)
3. **Τελικός score**: σταθμισμένος μέσος όρος + 10-bin clinical scoring

## Δομή project

```
Health Technology Final/
├── app/                      # Flask web app
│   ├── app.py
│   ├── templates/index.html
│   └── static/recorder.js   # Web Audio API recorder
├── src/                      # Reusable modules
│   ├── features.py          # parselmouth + librosa feature extraction
│   ├── preprocessing.py     # Audio normalization pipeline
│   ├── scoring.py           # 10-bin clinical scoring
│   └── explain.py           # SHAP per-sample explainability
├── notebooks/                # Reproducibility notebooks
│   ├── 00_preprocessing_visualization.ipynb
│   ├── 01_compare_extraction.ipynb
│   ├── 02_extract_features.ipynb
│   ├── 03_train_models.ipynb
│   ├── 04_retrain_robust.ipynb
│   ├── 05_balanced_uci.ipynb
│   ├── 06_iyer_native_sr.ipynb
│   ├── 07_add_svm.ipynb
│   └── 08_holdout_retrain.ipynb
├── models/                   # Trained .joblib models
└── data/                     # Datasets (audio gitignored, features kept)
```

## Setup

```bash
git clone <repo-url>
cd "Health Technology Final"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Datasets

Τα audio files είναι μεγάλα (~700MB) και έχουν δικές τους άδειες. Κατεβάζονται ξεχωριστά:

- **UCI**: https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification → `pd_speech_features.csv` στο `data/uci/`
- **Iyer 2023**: https://figshare.com/articles/dataset/Voice_Samples_for_Patients_with_Parkinson_s_Disease_and_Healthy_Controls/23849127 → unzip σε `data/iyer/`
- **MDVR-KCL**: https://zenodo.org/records/2867216 → unzip σε `data/mdvr_kcl/`

Τα **extracted features CSVs** (`iyer_features_8khz.csv`, `mdvr_features.csv`) **έχουν committed στο repo** ώστε να μπορείς να ξανατρέφεις μοντέλα χωρίς τα audio files.

## Εκτέλεση

```bash
python app/app.py
```

Άνοιξε http://localhost:5050

## Key technical findings

### 1. Sample rate mismatch (Iyer)
Iyer dataset ηχογραφήθηκε σε **8 kHz** (telephone quality). Browser audio είναι 44.1 kHz με 5x μεγαλύτερο spectrum. **Λύση**: downsample το user audio σε 8 kHz πριν περάσει στο Iyer μοντέλο.

### 2. HNR computation bug
Το default `parselmouth.Sound.to_harmonicity_ac()` (autocorrelation) δίνει αρνητικές τιμές HNR ακόμα και για καθαρό audio. **Λύση**: χρήση `to_harmonicity_cc()` (cross-correlation) που είναι κατάλληλο για sustained vowels.

### 3. UCI scale mismatch
Τα intensity features του UCI dataset extracted με άγνωστη Sakar protocol. Δεν αναπαράγεται 1-προς-1 με δικό μας extraction. **Λύση**: αφαίρεση intensity features από το UCI μοντέλο.

### 4. UCI class imbalance
564 PD vs 192 HC (74% PD bias). Standard training τείνει να προβλέπει PD. **Λύση**: subject-level undersampling σε 50/50 = 384 samples.

## Out-of-sample evaluation

30 holdout subjects (10/dataset, 5 HC + 5 PD) που δεν χρησιμοποιήθηκαν στο training:

- Iyer holdout: **60% accuracy** (matches CV)
- MDVR holdout: **65% accuracy** (matches CV)
- Overall: **63% accuracy** with 80% sensitivity (PD) and 47% specificity (HC)

Το σύστημα έχει **bias προς PD prediction** — αναμενόμενο σε datasets που έχουν περισσότερους PD από HC, ακόμα και μετά από balancing.

## Datasets references

1. Sakar, C.O., Serbes, G., et al. (2019). "A comparative analysis of speech signal processing algorithms for Parkinson's disease classification and the use of the tunable Q-factor wavelet transform." *Applied Soft Computing*, 74, 255-263.
2. Iyer, A., et al. (2023). *Voice Samples for Patients with Parkinson's Disease and Healthy Controls*. Figshare.
3. Jaeger, H., Trivedi, D., & Stadtschnitzer, M. (2019). *Mobile Device Voice Recordings at King's College London (MDVR-KCL)*. Zenodo. doi: 10.5281/zenodo.2867215

## License

Σχολική εργασία. Τα datasets διέπονται από τις δικές τους άδειες (βλ. παραπάνω links).
