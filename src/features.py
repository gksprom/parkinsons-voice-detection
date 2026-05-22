"""Feature extraction από raw WAV audio.

Αναπαράγει subset των UCI Parkinson features χρησιμοποιώντας parselmouth (Praat)
και librosa. Τα ονόματα των features ταιριάζουν ακριβώς με του UCI CSV ώστε
να μπορούμε να χρησιμοποιήσουμε μοντέλο εκπαιδευμένο στο UCI dataset.
"""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call


# Praat default parameters for jitter/shimmer (φωνήεν /a/, ενήλικες)
PERIOD_FLOOR = 0.0001
PERIOD_CEILING = 0.02
MAX_PERIOD_FACTOR = 1.3
MAX_AMPLITUDE_FACTOR = 1.6
PITCH_FLOOR = 75.0
PITCH_CEILING = 600.0


def _safe(fn, default=0.0):
    """Τρέξε function, επέστρεψε default αν επιστρέψει nan ή σπάσει."""
    try:
        v = fn()
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default


class AudioQualityError(Exception):
    """Raised όταν το audio δεν είναι κατάλληλο για analysis (σιωπή, πολύ μικρό κλπ)."""
    pass


def validate_audio(wav_path, min_duration=1.5, min_rms_db=-50, min_voiced_ratio=0.3):
    """Έλεγξε ότι το audio έχει αρκετό voiced speech.

    Raises AudioQualityError αν το audio είναι silent ή χωρίς voiced frames.
    """
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    duration = len(y) / sr

    if duration < min_duration:
        raise AudioQualityError(
            f'Πολύ μικρή ηχογράφηση ({duration:.1f}s). Χρειάζονται τουλάχιστον {min_duration}s.'
        )

    # RMS amplitude check (silence detection)
    rms = float(np.sqrt(np.mean(y ** 2)))
    rms_db = 20 * np.log10(rms + 1e-10)
    if rms_db < min_rms_db:
        raise AudioQualityError(
            f'Δεν εντοπίστηκε ήχος (RMS {rms_db:.1f} dB). Μίλα πιο δυνατά ή πιο κοντά στο μικρόφωνο.'
        )

    # Voiced frame ratio: πόσα frames έχουν detectable pitch
    sound = parselmouth.Sound(str(wav_path))
    pitch = sound.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    pitch_values = pitch.selected_array['frequency']
    voiced_ratio = float(np.sum(pitch_values > 0) / len(pitch_values)) if len(pitch_values) else 0
    if voiced_ratio < min_voiced_ratio:
        raise AudioQualityError(
            f'Δεν εντοπίστηκε αρκετή φωνή ({voiced_ratio*100:.0f}% voiced frames). '
            f'Πες ένα σταθερό "αααα" χωρίς διακοπές.'
        )

    return {'duration': duration, 'rms_db': rms_db, 'voiced_ratio': voiced_ratio}


def extract_features(wav_path):
    """Επιστρέφει dict με features από WAV αρχείο.

    Τα keys ταιριάζουν με τα column names του UCI pd_speech_features.csv.
    """
    sound = parselmouth.Sound(str(wav_path))

    # Pitch + PointProcess για jitter/shimmer
    pitch = sound.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    point_process = call(sound, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)

    features = {}

    # --- Pulse statistics ---
    features['numPulses'] = _safe(lambda: call(point_process, "Get number of points"))
    features['numPeriodsPulses'] = _safe(lambda: call(
        point_process, "Get number of periods", 0, 0,
        PERIOD_FLOOR, PERIOD_CEILING, MAX_PERIOD_FACTOR))
    features['meanPeriodPulses'] = _safe(lambda: call(
        point_process, "Get mean period", 0, 0,
        PERIOD_FLOOR, PERIOD_CEILING, MAX_PERIOD_FACTOR))
    features['stdDevPeriodPulses'] = _safe(lambda: call(
        point_process, "Get stdev period", 0, 0,
        PERIOD_FLOOR, PERIOD_CEILING, MAX_PERIOD_FACTOR))

    # --- Jitter (5) ---
    features['locPctJitter'] = _safe(lambda: call(
        point_process, "Get jitter (local)", 0, 0,
        PERIOD_FLOOR, PERIOD_CEILING, MAX_PERIOD_FACTOR))
    features['locAbsJitter'] = _safe(lambda: call(
        point_process, "Get jitter (local, absolute)", 0, 0,
        PERIOD_FLOOR, PERIOD_CEILING, MAX_PERIOD_FACTOR))
    features['rapJitter'] = _safe(lambda: call(
        point_process, "Get jitter (rap)", 0, 0,
        PERIOD_FLOOR, PERIOD_CEILING, MAX_PERIOD_FACTOR))
    features['ppq5Jitter'] = _safe(lambda: call(
        point_process, "Get jitter (ppq5)", 0, 0,
        PERIOD_FLOOR, PERIOD_CEILING, MAX_PERIOD_FACTOR))
    features['ddpJitter'] = _safe(lambda: call(
        point_process, "Get jitter (ddp)", 0, 0,
        PERIOD_FLOOR, PERIOD_CEILING, MAX_PERIOD_FACTOR))

    # --- Shimmer (6) ---
    features['locShimmer'] = _safe(lambda: call(
        [sound, point_process], "Get shimmer (local)", 0, 0,
        PERIOD_FLOOR, PERIOD_CEILING, MAX_PERIOD_FACTOR, MAX_AMPLITUDE_FACTOR))
    features['locDbShimmer'] = _safe(lambda: call(
        [sound, point_process], "Get shimmer (local_dB)", 0, 0,
        PERIOD_FLOOR, PERIOD_CEILING, MAX_PERIOD_FACTOR, MAX_AMPLITUDE_FACTOR))
    features['apq3Shimmer'] = _safe(lambda: call(
        [sound, point_process], "Get shimmer (apq3)", 0, 0,
        PERIOD_FLOOR, PERIOD_CEILING, MAX_PERIOD_FACTOR, MAX_AMPLITUDE_FACTOR))
    features['apq5Shimmer'] = _safe(lambda: call(
        [sound, point_process], "Get shimmer (apq5)", 0, 0,
        PERIOD_FLOOR, PERIOD_CEILING, MAX_PERIOD_FACTOR, MAX_AMPLITUDE_FACTOR))
    features['apq11Shimmer'] = _safe(lambda: call(
        [sound, point_process], "Get shimmer (apq11)", 0, 0,
        PERIOD_FLOOR, PERIOD_CEILING, MAX_PERIOD_FACTOR, MAX_AMPLITUDE_FACTOR))
    features['ddaShimmer'] = _safe(lambda: call(
        [sound, point_process], "Get shimmer (dda)", 0, 0,
        PERIOD_FLOOR, PERIOD_CEILING, MAX_PERIOD_FACTOR, MAX_AMPLITUDE_FACTOR))

    # --- Harmonicity (3) ---
    # CC (cross-correlation) αντί για AC: το AC default δίνει λάθος αρνητικές
    # τιμές HNR για clean audio. Το CC είναι το correct method για sustained vowels.
    harmonicity_ac = sound.to_harmonicity_cc()
    hnr_mean = _safe(lambda: call(harmonicity_ac, "Get mean", 0, 0))
    features['meanAutoCorrHarmonicity'] = hnr_mean / 10.0  # approx autocorr value
    features['meanHarmToNoiseHarmonicity'] = hnr_mean  # HNR in dB
    # NHR = 10^(-HNR/10) σε linear scale
    features['meanNoiseToHarmHarmonicity'] = float(10 ** (-hnr_mean / 10.0)) if hnr_mean > 0 else 0.0

    # --- Intensity (3) ---
    intensity = sound.to_intensity()
    features['meanIntensity'] = _safe(lambda: call(intensity, "Get mean", 0, 0, "dB"))
    features['minIntensity'] = _safe(lambda: call(intensity, "Get minimum", 0, 0, "Parabolic"))
    features['maxIntensity'] = _safe(lambda: call(intensity, "Get maximum", 0, 0, "Parabolic"))

    # --- Formants + Bandwidths (4 + 4) ---
    formants = sound.to_formant_burg(max_number_of_formants=5)
    duration = sound.get_total_duration()
    # Average over the middle 80% του signal
    times = np.linspace(0.1 * duration, 0.9 * duration, 20)
    for i in range(1, 5):
        f_vals = [call(formants, "Get value at time", i, t, "hertz", "linear") for t in times]
        b_vals = [call(formants, "Get bandwidth at time", i, t, "hertz", "linear") for t in times]
        f_vals = [v for v in f_vals if v and not np.isnan(v)]
        b_vals = [v for v in b_vals if v and not np.isnan(v)]
        features[f'f{i}'] = float(np.mean(f_vals)) if f_vals else 0.0
        features[f'b{i}'] = float(np.mean(b_vals)) if b_vals else 0.0

    # --- MFCCs (13 means + 13 stds) via librosa ---
    y, sr = librosa.load(str(wav_path), sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfcc.mean(axis=1)
    mfcc_stds = mfcc.std(axis=1)
    ordinals = ['0th', '1st', '2nd', '3rd', '4th', '5th', '6th',
                '7th', '8th', '9th', '10th', '11th', '12th']
    for i, ord_str in enumerate(ordinals):
        features[f'mean_MFCC_{ord_str}_coef'] = float(mfcc_means[i])
        features[f'std_MFCC_{ord_str}_coef'] = float(mfcc_stds[i])

    return features


# Λίστα με όλα τα feature names στη σωστή σειρά
FEATURE_NAMES = [
    'numPulses', 'numPeriodsPulses', 'meanPeriodPulses', 'stdDevPeriodPulses',
    'locPctJitter', 'locAbsJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter',
    'locShimmer', 'locDbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer',
    'meanAutoCorrHarmonicity', 'meanNoiseToHarmHarmonicity', 'meanHarmToNoiseHarmonicity',
    'minIntensity', 'maxIntensity', 'meanIntensity',
    'f1', 'f2', 'f3', 'f4', 'b1', 'b2', 'b3', 'b4',
] + [f'mean_MFCC_{o}_coef' for o in ['0th', '1st', '2nd', '3rd', '4th', '5th', '6th',
                                      '7th', '8th', '9th', '10th', '11th', '12th']] \
  + [f'std_MFCC_{o}_coef' for o in ['0th', '1st', '2nd', '3rd', '4th', '5th', '6th',
                                     '7th', '8th', '9th', '10th', '11th', '12th']]


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python features.py <wav_file>')
        sys.exit(1)
    feats = extract_features(sys.argv[1])
    for name in FEATURE_NAMES:
        print(f'{name}: {feats[name]:.4f}')
    print(f'\nTotal: {len(feats)} features')
