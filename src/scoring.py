"""10-bin scoring system για PD probability.

Από Shen et al. 2025 (Table 3). Κάθε bin έχει κλινική περιγραφή ώστε
ο user να καταλαβαίνει τι σημαίνει το score, όχι μόνο ένα νούμερο.
"""

SCORING_BINS = [
    (0.00, 0.15, 'Πολύ χαμηλή πιθανότητα',
     'Φωνητικά χαρακτηριστικά υγιούς προφίλ. Καμία ένδειξη Πάρκινσον.'),
    (0.15, 0.30, 'Χαμηλή πιθανότητα',
     'Μερικά χαρακτηριστικά ίσως ελαφρώς άτυπα, γενικά εντός υγιούς εύρους.'),
    (0.30, 0.45, 'Αβέβαιο, ελαφρά υπέρ HC',
     'Το μοντέλο γέρνει προς υγιές αλλά με αβεβαιότητα. Πιθανότατα φυσιολογικό.'),
    (0.45, 0.55, 'Αβέβαιο',
     'Το μοντέλο δεν είναι σίγουρο. Εντός του range όπου το μοντέλο έχει χαμηλή αξιοπιστία.'),
    (0.55, 0.65, 'Αβέβαιο, ελαφρά υπέρ PD',
     'Ορισμένα χαρακτηριστικά μοιάζουν με PD αλλά η πρόβλεψη είναι αβέβαιη. Το μοντέλο έχει 73% accuracy οπότε ψευδώς θετικά είναι συχνά.'),
    (0.65, 0.75, 'Μέτρια ένδειξη',
     'Αρκετά χαρακτηριστικά συμφωνούν με PD προφίλ. Αν συντρέχουν άλλα συμπτώματα, σκόπιμη κλινική αξιολόγηση.'),
    (0.75, 0.85, 'Υψηλή ένδειξη',
     'Τα περισσότερα χαρακτηριστικά συμφωνούν με PD. Συστήνεται κλινική αξιολόγηση.'),
    (0.85, 0.95, 'Πολύ υψηλή ένδειξη',
     'Σχεδόν όλα τα χαρακτηριστικά αντιστοιχούν σε γνωστά PD μοτίβα.'),
    (0.95, 1.01, 'Σχεδόν βέβαιη ένδειξη',
     'Τα χαρακτηριστικά πληρούν όλα ή σχεδόν όλα τα κλινικά κριτήρια για Πάρκινσον.'),
]


def score_to_bin(score):
    """Επιστρέφει το bin που πέφτει το score."""
    for low, high, label, desc in SCORING_BINS:
        if low <= score < high:
            return {
                'score': round(score, 3),
                'bin_low': low,
                'bin_high': high,
                'label': label,
                'description': desc,
            }
    return {
        'score': round(score, 3),
        'bin_low': 0.9,
        'bin_high': 1.0,
        'label': 'Definite likelihood',
        'description': SCORING_BINS[-1][3],
    }


def combine_scores(scores, weights=None):
    """Συνδυασμός πολλαπλών scores. Default = weighted mean.

    scores: list of (name, probability)
    weights: dict name -> weight (default ίσο βάρος)
    """
    if weights is None:
        weights = {name: 1.0 for name, _ in scores}

    total_w = sum(weights.get(n, 1.0) for n, _ in scores)
    combined = sum(p * weights.get(n, 1.0) for n, p in scores) / total_w
    return combined
