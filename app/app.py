"""Flask εφαρμογή με 3-model ensemble.

Ροή:
1. Step 1: Ηχογράφηση sustained vowel "αααα" -> UCI model + Iyer model (διπλό safety)
2. Step 2: Ηχογράφηση ανάγνωσης ελληνικού κειμένου -> MDVR model
3. Final: Combined probability με 10-bin scoring system
"""

import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, session

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import extract_features, validate_audio, AudioQualityError, FEATURE_NAMES as ALL_FEATURE_NAMES
from src.preprocessing import preprocess_audio, assess_recording_quality
from src.scoring import score_to_bin, combine_scores
from src.explain import ModelExplainer


app = Flask(__name__)
app.secret_key = 'health-tech-final-2025'  # για session

# Φόρτωση 3 μοντέλων με όλα τα 55 features
MODELS_DIR = ROOT / 'models'
DATA_DIR = ROOT / 'data'
# RF models — UCI χωρίς intensity features (scale mismatch με δικό μας extraction)
uci_model_rf = joblib.load(MODELS_DIR / 'uci_balanced_noint.joblib')
iyer_model_rf = joblib.load(MODELS_DIR / 'iyer_8khz.joblib')
mdvr_model_rf = joblib.load(MODELS_DIR / 'mdvr_model.joblib')

# SVM models
uci_model_svm = joblib.load(MODELS_DIR / 'uci_balanced_noint_svm.joblib')
iyer_model_svm = joblib.load(MODELS_DIR / 'iyer_8khz_svm.joblib')
mdvr_model_svm = joblib.load(MODELS_DIR / 'mdvr_svm.joblib')

# Feature subset για UCI (52 features, χωρίς intensity)
UCI_FEATURES = joblib.load(MODELS_DIR / 'uci_clean_features.joblib')

# Aliases για backward compat
uci_model = uci_model_rf
iyer_model = iyer_model_rf
mdvr_model = mdvr_model_rf

FEATURE_NAMES = ALL_FEATURE_NAMES
print(f'Loaded 6 models (3 RF + 3 SVM), {len(FEATURE_NAMES)} features each')

# Build SHAP explainers με training data
print('Building SHAP explainers...')
uci_df = pd.read_csv(DATA_DIR / 'uci' / 'pd_speech_features.csv', header=1)
uci_explainer = ModelExplainer(uci_model, uci_df[UCI_FEATURES].values,
                                uci_df['class'].values, UCI_FEATURES)

iyer_df = pd.read_csv(DATA_DIR / 'iyer' / 'iyer_features_8khz.csv')
iyer_explainer = ModelExplainer(iyer_model, iyer_df[FEATURE_NAMES].values,
                                 iyer_df['class'].values, FEATURE_NAMES)

mdvr_df = pd.read_csv(DATA_DIR / 'mdvr_kcl' / 'mdvr_features.csv')
mdvr_explainer = ModelExplainer(mdvr_model, mdvr_df[FEATURE_NAMES].values,
                                 mdvr_df['class'].values, FEATURE_NAMES)
print('SHAP explainers ready')

# Reading passage για Part 2 — απόσπασμα από "The North Wind and the Sun"
# (classic phonetic passage που χρησιμοποιείται σε speech research, IPA standard).
GREEK_READING_TEXT = (
    'The North Wind and the Sun were disputing which was the stronger, '
    'when a traveler came along wrapped in a warm cloak. '
    'They agreed that the one who first succeeded in making the traveler '
    'take off his cloak should be considered stronger than the other.'
)


def predict_from_wav(wav_path, models, sample_rates=None, feature_sets=None):
    """Preprocess + extract features + predict με κάθε model.

    sample_rates: dict {model_name: target_sr}. Downsample audio per model.
    feature_sets: dict {model_name: feature_list}. Subset of FEATURE_NAMES.
    """
    import librosa
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        preprocess_audio(wav_path, output_path=tmp.name)
        validate_audio(tmp.name)
        default_features = extract_features(tmp.name)

    results = {}
    for name, model in models.items():
        target_sr = sample_rates.get(name) if sample_rates else None
        feature_list = (feature_sets or {}).get(name, FEATURE_NAMES)

        if target_sr and target_sr != 44100:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_sr:
                y, _ = librosa.load(tmp.name, sr=target_sr, mono=True)
                sf.write(tmp_sr.name, y, target_sr)
                feats = extract_features(tmp_sr.name)
                Path(tmp_sr.name).unlink(missing_ok=True)
        else:
            feats = default_features

        x = pd.DataFrame([[feats[fname] for fname in feature_list]], columns=feature_list)
        proba = model.predict_proba(x)[0][1]
        results[name] = float(proba)

    Path(tmp.name).unlink(missing_ok=True)
    return results, default_features


@app.route('/')
def index():
    return render_template('index.html', reading_text=GREEK_READING_TEXT)


@app.route('/predict/vowel', methods=['POST'])
def predict_vowel():
    """Step 1: Sustained vowel /a/ -> UCI + Iyer."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio uploaded'}), 400

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        request.files['audio'].save(tmp.name)
        wav_path = tmp.name

    try:
        # Recording quality πρώτα (πριν το preprocess για να δει το raw input)
        quality = assess_recording_quality(wav_path)

        # UCI 44.1 kHz χωρίς intensity, Iyer 8 kHz με όλα τα features
        probs, feats = predict_from_wav(
            wav_path,
            {
                'uci_rf': uci_model_rf, 'uci_svm': uci_model_svm,
                'iyer_rf': iyer_model_rf, 'iyer_svm': iyer_model_svm,
            },
            sample_rates={
                'uci_rf': 44100, 'uci_svm': 44100,
                'iyer_rf': 8000, 'iyer_svm': 8000,
            },
            feature_sets={
                'uci_rf': UCI_FEATURES, 'uci_svm': UCI_FEATURES,
            },
        )

        # Ensemble RF+SVM ανά dataset (απλός μέσος)
        probs['uci'] = (probs['uci_rf'] + probs['uci_svm']) / 2
        probs['iyer'] = (probs['iyer_rf'] + probs['iyer_svm']) / 2

        # UCI BALANCED (undersampled to 50/50) + Iyer (50/50) με ίσο βάρος.
        # Το balanced UCI έχει 69% accuracy αλλά balanced HC/PD recall (69%/70%)
        # σε αντίθεση με το imbalanced που είχε φαινομενικά υψηλό accuracy αλλά
        # HC recall μόνο 32%.
        vowel_combined = 0.5 * probs['uci'] + 0.5 * probs['iyer']

        # Model agreement (μεταξύ UCI και Iyer, για info μόνο)
        agreement = 1 - abs(probs['uci'] - probs['iyer'])

        # SHAP explanations
        uci_top = uci_explainer.explain(feats, top_n=5)
        iyer_top = iyer_explainer.explain(feats, top_n=5)

        # Σώσε στο session για το final
        session['vowel'] = {
            'uci_prob': probs['uci'],
            'uci_rf_prob': probs['uci_rf'],
            'uci_svm_prob': probs['uci_svm'],
            'iyer_prob': probs['iyer'],
            'iyer_rf_prob': probs['iyer_rf'],
            'iyer_svm_prob': probs['iyer_svm'],
            'combined': vowel_combined,
        }

        return jsonify({
            'uci_prob': round(probs['uci'], 3),
            'uci_rf_prob': round(probs['uci_rf'], 3),
            'uci_svm_prob': round(probs['uci_svm'], 3),
            'iyer_prob': round(probs['iyer'], 3),
            'iyer_rf_prob': round(probs['iyer_rf'], 3),
            'iyer_svm_prob': round(probs['iyer_svm'], 3),
            'combined': round(vowel_combined, 3),
            'agreement': round(agreement, 3),
            'bin': score_to_bin(vowel_combined),
            'uci_top_features': uci_top,
            'iyer_top_features': iyer_top,
            'quality': quality,
        })
    except AudioQualityError as e:
        return jsonify({'error': str(e), 'audio_invalid': True}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        Path(wav_path).unlink(missing_ok=True)


@app.route('/predict/reading', methods=['POST'])
def predict_reading():
    """Step 2: Reading text -> MDVR."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio uploaded'}), 400

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        request.files['audio'].save(tmp.name)
        wav_path = tmp.name

    try:
        probs, feats = predict_from_wav(wav_path, {
            'mdvr_rf': mdvr_model_rf, 'mdvr_svm': mdvr_model_svm,
        })
        probs['mdvr'] = (probs['mdvr_rf'] + probs['mdvr_svm']) / 2

        session['reading'] = {
            'mdvr_prob': probs['mdvr'],
        }

        # Combine vowel + reading για final
        if 'vowel' in session:
            vowel_combined = session['vowel']['combined']
            # Παρατήρηση: MDVR παρουσιάζει σωστό clinical HNR pattern (HC > PD).
            # Iyer έχει inverse HNR (HC<PD, dataset bias).
            # MDVR είναι πιο αξιόπιστο για κλινική νοοτροπία, οπότε 50/50.
            final = vowel_combined * 0.5 + probs['mdvr'] * 0.5

            # All-models agreement
            uci_p = session['vowel']['uci_prob']
            iyer_p = session['vowel']['iyer_prob']
            mdvr_p = probs['mdvr']
            spread = max(uci_p, iyer_p, mdvr_p) - min(uci_p, iyer_p, mdvr_p)
            agreement = 1 - spread
        else:
            final = probs['mdvr']
            agreement = 1.0

        session['final'] = final

        # SHAP explanation για MDVR
        mdvr_top = mdvr_explainer.explain(feats, top_n=5)
        quality = assess_recording_quality(wav_path)

        return jsonify({
            'mdvr_prob': round(probs['mdvr'], 3),
            'mdvr_rf_prob': round(probs['mdvr_rf'], 3),
            'mdvr_svm_prob': round(probs['mdvr_svm'], 3),
            'final': round(final, 3),
            'agreement': round(agreement, 3),
            'bin': score_to_bin(final),
            'breakdown': {
                'uci_prob': round(session.get('vowel', {}).get('uci_prob', 0), 3),
                'uci_rf_prob': round(session.get('vowel', {}).get('uci_rf_prob', 0), 3),
                'uci_svm_prob': round(session.get('vowel', {}).get('uci_svm_prob', 0), 3),
                'iyer_prob': round(session.get('vowel', {}).get('iyer_prob', 0), 3),
                'iyer_rf_prob': round(session.get('vowel', {}).get('iyer_rf_prob', 0), 3),
                'iyer_svm_prob': round(session.get('vowel', {}).get('iyer_svm_prob', 0), 3),
                'mdvr_prob': round(probs['mdvr'], 3),
                'vowel_avg': round(session.get('vowel', {}).get('combined', 0), 3),
            },
            'mdvr_top_features': mdvr_top,
            'quality': quality,
        })
    except AudioQualityError as e:
        return jsonify({'error': str(e), 'audio_invalid': True}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        Path(wav_path).unlink(missing_ok=True)


@app.route('/reset', methods=['POST'])
def reset():
    session.clear()
    return jsonify({'ok': True})


if __name__ == '__main__':
    app.run(debug=True, port=5050)
