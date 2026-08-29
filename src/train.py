"""
Module d'entraînement des modèles de détection de fraude bancaire.

Compare trois algorithmes avec gestion du déséquilibre intégrée :
- LogisticRegression  (class_weight="balanced")
- RandomForestClassifier (class_weight="balanced")
- XGBClassifier       (scale_pos_weight ajusté automatiquement)

Chaque modèle est sauvegardé au format joblib dans le dossier models/.
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    prepare_full_pipeline,
)

FEATURES = ["Time"] + FEATURE_COLUMNS + ["Amount"]

MODELS_DIR = "models"


def _compute_ratio(y_train):
    """
    Calcule le ratio de déséquilibre pour XGBoost : nombre de produits
    divisé par nombre de fraudes. Utile pour scale_pos_weight.
    """
    n_total = len(y_train)
    n_fraudes = int((y_train == 1).sum())
    n_normaux = n_total - n_fraudes
    return n_normaux / max(n_fraudes, 1)


def build_logistic_regression():
    """
    Régression logistique avec stratégie de régularisation augmentée et
    gestion du déséquilibre via class_weight="balanced".

    Le paramètre class_weight="balanced" pondère automatiquement les
    classes inversement à leur fréquence.
    """
    return LogisticRegression(
        class_weight="balanced",
        max_iter=2000,  # le solvant a besoin de plus d'itérations ici
        random_state=42,
        solver="liblinear",
    )


def build_random_forest():
    """
    Forêt aléatoire avec class_weight="balanced".
    Elle gère naturellement les interactions non linéaires entre les
    features (temps, montant, composantes V1-V28).
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def build_xgboost(y_train):
    """
    XGBoost avec gestion du déséquilibre via scale_pos_weight
    (= ratio normal/fraude), ce qui est équivalent à un class_weight
    côté gradient boosting.

    Le paramètre eval_metric est fixé à une métrique robuste au
    déséquilibre (logloss).
    """
    scale_pos_weight = _compute_ratio(y_train)
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,  # gestion du déséquilibre
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def train_model(name, model, X_train, y_train):
    """
    Entraîne un modèle et mesure ses performances rapides sur le jeu
    d'entraînement (aucun test ici : l'évaluation complète est faite
    dans les notebooks).
    """
    print(f"\n--- Entraînement : {name} ---")
    model.fit(X_train, y_train)

    # Petites métriques de contrôle sur l'apprentissage (suroptimisation attendue)
    proba = model.predict_proba(X_train)[:, 1]
    print(f"  ROC-AUC (train) : {roc_auc_score(y_train, proba):.4f}")
    print(f"  PR-AUC  (train) : {average_precision_score(y_train, proba):.4f}")
    return model


def save_model(model, name):
    """
    Sauvegarde un modèle entraîné au format joblib, accompagné de la liste
    des features attendues en entrée (utile pour la prédiction).
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.joblib")

    # Bundle : modèle + ordre des features + cible
    joblib.dump(
        {"model": model, "features": FEATURES, "target": TARGET_COLUMN},
        path,
    )
    print(f"  Modèle sauvegardé dans {path}")
    return path


def train_all(path="data/creditcard.csv", use_smote=False):
    """
    Orchestre l'entraînement des trois modèles.

    Paramètres
    ----------
    path : str
        Chemin vers creditcard.csv.
    use_smote : bool
        Applique SMOTE avant l'entraînement (désequilibrage renforcé).

    Retourne
    --------
    (models, data) avec models = dict nom -> modèle entraîné
    et data = dictionnaire du pipeline de préprocessing.
    """
    # 1. Préparation des données
    data = prepare_full_pipeline(
        path=path,
        test_size=0.2,
        random_state=42,
        use_smote=use_smote,
    )
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    # 2. Construction des modèles
    models = {
        "logistic_regression": build_logistic_regression(),
        "random_forest": build_random_forest(),
        "xgboost": build_xgboost(y_train),
    }

    # 3. Entraînement + sauvegarde
    trained = {}
    for name, model in models.items():
        model = train_model(name, model, X_train, y_train)
        save_model(model, name)
        trained[name] = model

    return trained, data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entraîne les modèles de détection de fraude bancaire."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/creditcard.csv",
        help="Chemin vers le fichier CSV de données.",
    )
    parser.add_argument(
        "--smote",
        action="store_true",
        help="Applique SMOTE pour rééquilibrer le jeu d'entraînement.",
    )
    args = parser.parse_args()

    train_all(path=args.data, use_smote=args.smote)