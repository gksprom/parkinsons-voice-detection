"""Per-sample explainability με SHAP + ελληνικά labels.

Για κάθε prediction επιστρέφει top features που έσπρωξαν την απόφαση
προς PD ή HC, μαζί με σύγκριση με τα training distributions HC/PD.
"""

import numpy as np
import pandas as pd
import shap


# Ελληνικά ονόματα για τα features
FEATURE_LABELS_GR = {
    'numPulses': 'Αριθμός φωνητικών παλμών',
    'numPeriodsPulses': 'Αριθμός περιόδων ταλάντωσης',
    'meanPeriodPulses': 'Μέση περίοδος ταλάντωσης',
    'stdDevPeriodPulses': 'Διακύμανση περιόδου',

    'locPctJitter': 'Jitter (local %) – αστάθεια συχνότητας',
    'locAbsJitter': 'Jitter (absolute) – απόλυτη αστάθεια',
    'rapJitter': 'RAP Jitter – σχετική σταθερότητα',
    'ppq5Jitter': 'PPQ5 Jitter – 5-σημείο pitch variability',
    'ddpJitter': 'DDP Jitter – cycle-to-cycle διαφορά',

    'locShimmer': 'Shimmer (local) – αστάθεια έντασης',
    'locDbShimmer': 'Shimmer (dB) – διακύμανση πλάτους',
    'apq3Shimmer': 'APQ3 Shimmer',
    'apq5Shimmer': 'APQ5 Shimmer',
    'apq11Shimmer': 'APQ11 Shimmer',
    'ddaShimmer': 'DDA Shimmer',

    'meanAutoCorrHarmonicity': 'Αυτοσυσχέτιση φωνής',
    'meanNoiseToHarmHarmonicity': 'Λόγος θορύβου προς αρμονικές (NHR)',
    'meanHarmToNoiseHarmonicity': 'Λόγος αρμονικών προς θόρυβο (HNR)',

    'minIntensity': 'Ελάχιστη ένταση φωνής (dB)',
    'maxIntensity': 'Μέγιστη ένταση φωνής (dB)',
    'meanIntensity': 'Μέση ένταση φωνής (dB)',

    'f1': '1ος formant (F1)',
    'f2': '2ος formant (F2)',
    'f3': '3ος formant (F3)',
    'f4': '4ος formant (F4)',
    'b1': 'Bandwidth F1',
    'b2': 'Bandwidth F2',
    'b3': 'Bandwidth F3',
    'b4': 'Bandwidth F4',
}

# MFCC labels
for i, ord_str in enumerate(['0th', '1st', '2nd', '3rd', '4th', '5th', '6th',
                              '7th', '8th', '9th', '10th', '11th', '12th']):
    FEATURE_LABELS_GR[f'mean_MFCC_{ord_str}_coef'] = f'Mean MFCC-{i} (φασματικό χαρ.)'
    FEATURE_LABELS_GR[f'std_MFCC_{ord_str}_coef'] = f'Std MFCC-{i} (διακύμανση φάσματος)'


def get_label(feature_name):
    return FEATURE_LABELS_GR.get(feature_name, feature_name)


class ModelExplainer:
    """Wraps a sklearn Pipeline (scaler + RF) με SHAP + training stats."""

    def __init__(self, pipeline, X_train, y_train, feature_names):
        self.pipeline = pipeline
        self.feature_names = feature_names

        # Scaler + classifier
        self.scaler = pipeline.named_steps['scaler']
        self.clf = pipeline.named_steps['clf']

        # SHAP explainer πάνω στον classifier (μετά το scaling)
        X_scaled = self.scaler.transform(X_train)
        self.explainer = shap.TreeExplainer(self.clf, X_scaled)

        # Training stats ανά class (HC, PD)
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        self.hc_stats = X_train_df[y_train == 0].agg(['mean', 'std']).T
        self.pd_stats = X_train_df[y_train == 1].agg(['mean', 'std']).T

    def explain(self, x_input, top_n=5):
        """Επιστρέφει top_n features με SHAP impact + comparison με HC/PD μέσους όρους.

        x_input: dict feature_name -> value
        """
        x_df = pd.DataFrame([[x_input[f] for f in self.feature_names]],
                            columns=self.feature_names)
        x_scaled = self.scaler.transform(x_df)

        # SHAP values για την class PD (1)
        shap_values = self.explainer.shap_values(x_scaled)
        if isinstance(shap_values, list):
            # Multi-class output: [class_0_shap, class_1_shap]
            shap_vec = shap_values[1][0]
        elif shap_values.ndim == 3:
            # New SHAP API: (n_samples, n_features, n_classes)
            shap_vec = shap_values[0, :, 1]
        else:
            shap_vec = shap_values[0]

        # Sort by absolute impact
        importance = np.abs(shap_vec)
        top_idx = np.argsort(importance)[-top_n:][::-1]

        results = []
        for i in top_idx:
            fname = self.feature_names[i]
            user_val = float(x_input[fname])
            hc_mean = float(self.hc_stats.loc[fname, 'mean'])
            pd_mean = float(self.pd_stats.loc[fname, 'mean'])
            shap_val = float(shap_vec[i])

            # Direction: positive shap = sprwxnei pros PD, negative = pros HC
            direction = 'PD' if shap_val > 0 else 'HC'

            # Πιο κοντά σε HC ή PD μέσο όρο?
            d_hc = abs(user_val - hc_mean)
            d_pd = abs(user_val - pd_mean)
            closer_to = 'PD' if d_pd < d_hc else 'HC'

            results.append({
                'feature': fname,
                'label_gr': get_label(fname),
                'user_value': round(user_val, 4),
                'hc_mean': round(hc_mean, 4),
                'pd_mean': round(pd_mean, 4),
                'shap_impact': round(shap_val, 4),
                'direction': direction,  # προς ποια class σπρώχνει
                'closer_to': closer_to,  # πιο κοντά σε ποιο μέσο όρο
            })

        return results
