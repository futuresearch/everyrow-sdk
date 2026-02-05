import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score


class TextClassifier:
    """Text classifier using TF-IDF + LightGBM.

    The vectorizer should be fit once on the full corpus (labeled + unlabeled)
    to ensure consistent features across iterations.
    """

    def __init__(self, vectorizer: TfidfVectorizer | None = None):
        """Initialize classifier.

        Args:
            vectorizer: Pre-fitted TfidfVectorizer. If None, will fit on first call.
        """
        self.vectorizer = vectorizer or TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2)
        )
        self.model = LGBMClassifier(
            n_estimators=100,
            num_leaves=31,
            learning_rate=0.1,
            verbose=-1,
        )
        self._vectorizer_fitted = vectorizer is not None
        self._fitted = False

    def fit(self, texts: list[str], labels: list[int]):
        """Fit the classifier on labeled data.

        If vectorizer wasn't pre-fitted, fits it here (not recommended for AL).
        """
        if not self._vectorizer_fitted:
            X = self.vectorizer.fit_transform(texts)
            self._vectorizer_fitted = True
        else:
            X = self.vectorizer.transform(texts)
        self.model.fit(X, labels)
        self._fitted = True

    def predict(self, texts: list[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)


def create_vectorizer(corpus: list[str]) -> TfidfVectorizer:
    """Create and fit a TF-IDF vectorizer on the given corpus.

    This should be called once at the start of the experiment with training
    texts only to avoid leakage from the test set.
    """
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    vectorizer.fit(corpus)
    return vectorizer


def train_classifier(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    vectorizer: TfidfVectorizer | None = None,
) -> tuple[TextClassifier, dict]:
    """Train a classifier and evaluate on test set.

    Args:
        train_df: Training data with 'text' and 'label' columns
        test_df: Test data with 'text' and 'label' columns
        vectorizer: Pre-fitted vectorizer (recommended for AL consistency)

    Returns:
        Tuple of (classifier, metrics_dict)
    """
    classifier = TextClassifier(vectorizer=vectorizer)

    classifier.fit(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
    )

    pred_labels = classifier.predict(test_df["text"].tolist())
    true_labels = test_df["label"].tolist()

    metrics = {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "f1_macro": float(f1_score(true_labels, pred_labels, average="macro")),
    }

    return classifier, metrics


def get_uncertainty_scores(classifier: TextClassifier, texts: list[str]) -> np.ndarray:
    """Compute entropy-based uncertainty scores for texts."""
    probs = classifier.predict_proba(texts)
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    return entropy
