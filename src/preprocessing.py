"""Audio preprocessing pipeline.

Εμπνευσμένο από τη μεθοδολογία των Shen et al. 2025 και Iyer et al. 2023:
1. Resample σε 44.1 kHz (αν χρειάζεται)
2. Convert σε mono
3. Noise reduction (spectral gating)
4. Silence trimming
5. RMS normalization σε σταθερό dB level
6. Peak rescale σε [-1, 1]

Σκοπός: μειώνει το domain gap μεταξύ training data (clinical) και live audio
(browser mic) ώστε το ίδιο extraction να δίνει συγκρίσιμα features.
"""

import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from pathlib import Path


TARGET_SR = 44100
TARGET_RMS_DB = -25.0   # ~normal speech level
TRIM_TOP_DB = 25         # silence threshold relative to peak
NOISE_DURATION = 0.5     # first N seconds used as noise sample


def preprocess_audio(
    wav_path,
    output_path=None,
    target_sr=TARGET_SR,
    target_rms_db=TARGET_RMS_DB,
    trim_top_db=TRIM_TOP_DB,
    apply_noise_reduction=True,
    return_array=False,
):
    """Preprocess WAV file. Επιστρέφει path του processed αρχείου ή numpy array.

    Steps:
    1. Load και resample σε target_sr, force mono
    2. Noise reduction (spectral gating με αυτόματη ανίχνευση σιωπηλών frames)
    3. Silence trim (start + end, ευρετική με top_db)
    4. RMS normalization σε target_rms_db
    5. Peak clip στο διάστημα [-1, 1]
    6. Αποθήκευση ως WAV
    """
    y, sr = librosa.load(str(wav_path), sr=target_sr, mono=True)

    if len(y) == 0:
        raise ValueError(f'Empty audio: {wav_path}')

    # 1. Aggressive 2-pass noise reduction για την αφαίρεση του mic noise floor
    if apply_noise_reduction:
        try:
            # Pass 1: αυτόματη ανίχνευση σιωπής και stationary noise reduction
            intervals = librosa.effects.split(y, top_db=20, frame_length=2048, hop_length=512)
            silent_mask = np.ones(len(y), dtype=bool)
            for start, end in intervals:
                silent_mask[start:end] = False
            noise_sample = y[silent_mask]

            if len(noise_sample) > sr * 0.15:
                y = nr.reduce_noise(
                    y=y, sr=sr, y_noise=noise_sample,
                    stationary=True, prop_decrease=0.9,
                )

            # Pass 2: non-stationary για residual θόρυβο
            y = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.6)
        except Exception as e:
            print(f'Noise reduction failed, skipping: {e}')

    # 2. Silence trim (start + end)
    y, _ = librosa.effects.trim(y, top_db=trim_top_db)

    if len(y) < sr * 0.3:
        raise ValueError(f'Πολύ μικρό audio μετά το trim ({len(y)/sr:.2f}s): {wav_path}')

    # 3. RMS normalization
    current_rms = np.sqrt(np.mean(y ** 2))
    if current_rms > 1e-6:
        target_rms_linear = 10 ** (target_rms_db / 20)
        gain = target_rms_linear / current_rms
        y = y * gain

    # 4. Peak clip σε [-1, 1]
    peak = np.max(np.abs(y))
    if peak > 1.0:
        y = y / peak

    if return_array:
        return y, sr

    if output_path is None:
        output_path = Path(wav_path).with_suffix('.processed.wav')

    sf.write(str(output_path), y, sr, subtype='PCM_16')
    return Path(output_path)


def preprocess_to_temp(wav_path):
    """Convenience: preprocess στο /tmp/processed.wav, επιστρέφει το path."""
    import tempfile
    out = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    out.close()
    return preprocess_audio(wav_path, output_path=out.name)


def assess_recording_quality(wav_path):
    """Επιστρέφει quality metrics + warnings για την ηχογράφηση.

    Σκοπός: να ξέρει ο user αν η πρόβλεψη είναι αξιόπιστη ή το mic έχει πρόβλημα.
    """
    import parselmouth
    from parselmouth.praat import call

    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    duration = len(y) / sr
    rms = np.sqrt(np.mean(y ** 2))
    rms_db = 20 * np.log10(rms + 1e-10)

    # HNR από parselmouth
    sound = parselmouth.Sound(str(wav_path))
    try:
        harm = sound.to_harmonicity_cc()  # CC method (correct for sustained vowels)
        hnr_db = float(call(harm, 'Get mean', 0, 0))
        if np.isnan(hnr_db):
            hnr_db = -10
    except Exception:
        hnr_db = -10

    # Voiced ratio
    try:
        pitch = sound.to_pitch(pitch_floor=75, pitch_ceiling=600)
        pv = pitch.selected_array['frequency']
        voiced_ratio = float(np.sum(pv > 0) / len(pv)) if len(pv) else 0
    except Exception:
        voiced_ratio = 0

    # Quality score (0-100)
    score = 100
    warnings = []

    if hnr_db < 5:
        score -= 40
        warnings.append({
            'level': 'critical',
            'text': f'Πολύ χαμηλό HNR ({hnr_db:.1f} dB). Η φωνή χάνεται μέσα στο θόρυβο. '
                    'Συνιστάται πιο ήσυχο περιβάλλον ή πιο κοντινή τοποθέτηση του μικροφώνου.'
        })
    elif hnr_db < 10:
        score -= 20
        warnings.append({
            'level': 'warning',
            'text': f'Σχετικά χαμηλό HNR ({hnr_db:.1f} dB). Το αποτέλεσμα μπορεί να επηρεαστεί από θόρυβο.'
        })

    if rms_db < -35:
        score -= 25
        warnings.append({
            'level': 'critical',
            'text': f'Πολύ χαμηλή ένταση φωνής ({rms_db:.1f} dB RMS). Απαιτείται μεγαλύτερη ένταση ή πιο κοντινή τοποθέτηση του μικροφώνου.'
        })
    elif rms_db < -28:
        score -= 10
        warnings.append({
            'level': 'warning',
            'text': f'Σχετικά σιγανή φωνή ({rms_db:.1f} dB).'
        })

    if voiced_ratio < 0.5:
        score -= 15
        warnings.append({
            'level': 'warning',
            'text': f'Λίγη ομιλία ({voiced_ratio*100:.0f}% voiced frames). Απαιτείται συνεχής εκφώνηση.'
        })

    if duration < 3:
        score -= 10
        warnings.append({
            'level': 'warning',
            'text': f'Σύντομη ηχογράφηση ({duration:.1f}s). Προτείνεται >5 sec.'
        })

    score = max(0, min(100, score))

    if score >= 75:
        verdict = 'good'
    elif score >= 50:
        verdict = 'fair'
    else:
        verdict = 'poor'

    return {
        'score': int(score),
        'verdict': verdict,
        'metrics': {
            'duration': float(round(duration, 2)),
            'rms_db': float(round(rms_db, 2)),
            'hnr_db': float(round(hnr_db, 2)),
            'voiced_ratio': float(round(voiced_ratio, 2)),
        },
        'warnings': warnings,
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python preprocessing.py <input.wav> [output.wav]')
        sys.exit(1)
    out = preprocess_audio(sys.argv[1], output_path=sys.argv[2] if len(sys.argv) > 2 else None)
    print(f'Saved: {out}')
