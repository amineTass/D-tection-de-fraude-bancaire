"""
Module de prétraitement des données pour la détection de fraude bancaire.

Ce module regroupe toutes les fonctions nécessaires au chargement,
au nettoyage, à la normalisation et à la répartition des données,
ainsi qu'à la gestion du déséquilibre des classes (SMOTE).

Le jeu de données attendu est celui de Kaggle : "Credit Card Fraud Detection"
(colonnes Time, V1-V28, Amount, Class).
"""

import os

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Colonnes du jeu de données Kaggle
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)]  # V1 -> V28
TARGET_COLUMN = "Class"
NON_FEATURE_COLUMNS = ["Time", "Amount", TARGET_COLUMN]


def load_data(path="data/creditcard.csv"):
    """
    Charge le jeu de données depuis un fichier CSV.

    Paramètres
    ----------
    path : str
        Chemin vers le fichier CSV contenant les transactions.

    Retourne
    --------
    pd.DataFrame
        DataFrame brut avec toutes les colonnes du jeu de données.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier introuvable : {path}. "
            "Téléchargez le dataset Kaggle 'Credit Card Fraud Detection' "
            "et placez-le dans le dossier data/."
        )
    df = pd.read_csv(path)
    print(f"Données chargées : {df.shape[0]} transactions, {df.shape[1]} colonnes.")
    return df


def scale_features(df, fit=True, scaler=None, save_path=None, load_path=None):
    """
    Normalise les colonnes 'Amount' et 'Time' avec un StandardScaler.

    Les colonnes V1-V28 sont déjà anonymisées et dimensionnées ; elles
    ne sont donc pas modifiées.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant au moins les colonnes Time et Amount.
    fit : bool
        Si True, ajuste le scaler sur les données ; sinon utilise un
        scaler fourni (utile en prédiction pour reproduire l'entraînement).
    scaler : StandardScaler, optionnel
        Scaler déjà ajusté, utilisé lorsque fit=False.
    save_path : str, optionnel
        Chemin où sauvegarder le scaler entraîné (joblib).
    load_path : str, optionnel
        Chemin d'un scaler sauvegardé à charger.

    Retourne
    --------
    (pd.DataFrame, StandardScaler)
        DataFrame avec Time et Amount standardisées + scaler utilisé.
    """
    df = df.copy()

    if load_path is not None:
        scaler = joblib.load(load_path)
        fit = False

    if scaler is None:
        scaler = StandardScaler()

    # On standardise les colonnes numériques à forte hétérogénéité
    scaler_cols = ["Time", "Amount"]
    if fit:
        df[scaler_cols] = scaler.fit_transform(df[scaler_cols])
    else:
        df[scaler_cols] = scaler.transform(df[scaler_cols])

    if save_path is not None:
        # On s'assure que le dossier de destination existe
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)
        print(f"Scaler sauvegardé dans : {save_path}")

    return df, scaler


def split_data(df, target=TARGET_COLUMN, test_size=0.2, random_state=42):
    """
    Sépare les données en jeux d'entraînement et de test de façon
    stratifiée (pour préserver la proportion des classes).

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant les features et la cible.
    target : str
        Nom de la colonne cible.
    test_size : float
        Proportion du jeu de test (défaut 0.2).
    random_state : int
        Graine aléatoire pour la reproductibilité.

    Retourne
    --------
    (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # stratification : même % de fraude dans chaque sous-ensemble
    )
    print(
        f"Split : {X_train.shape[0]} en entraînement, "
        f"{X_test.shape[0]} en test."
    )
    return X_train, X_test, y_train, y_test


def balance_data(X_train, y_train, random_state=42, sampling_strategy="auto"):
    """
    Rééquilibre le jeu d'entraînement avec SMOTE (suroveture synthétique
    de la classe minoritaire : les fraudes).

    Paramètres
    ----------
    X_train : pd.DataFrame
        Features d'entraînement.
    y_train : pd.Series
        Cible d'entraînement.
    random_state : int
        Graine aléatoire pour la reproductibilité.
    sampling_strategy : str | dict
        Stratégie de suréchantillonnage de SMOTE.

    Retourne
    --------
    (X_resampled, y_resampled)
    """
    n_fraudes = (y_train == 1).sum()
    if n_fraudes == 0:
        raise ValueError("Aucun exemple de fraude dans le jeu d'entraînement.")

    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(
        f"Rééquilibrage SMOTE : {X_train.shape[0]} -> {X_resampled.shape[0]} "
        f"exemples. Fraudes : {n_fraudes} -> {(y_resampled == 1).sum()}."
    )
    return X_resampled, y_resampled


def prepare_full_pipeline(
    path="data/creditcard.csv",
    test_size=0.2,
    random_state=42,
    use_smote=False,
    scaler_path="models/scaler.joblib",
):
    """
    Pipeline complet : chargement, normalisation, split et (optionnel) SMOTE.

    Paramètres
    ----------
    path : str
        Chemin du fichier CSV.
    test_size : float
        Taille du jeu de test.
    random_state : int
        Graine de reproductibilité.
    use_smote : bool
        Applique SMOTE sur le jeu d'entraînement si True.
    scaler_path : str
        Emplacement où sauvegarder le scaler entraîné.

    Retourne
    --------
    dict
        Clés : X_train, X_test, y_train, y_test, scaler.
    """
    df = load_data(path)

    # Normalisation de Time et Amount
    df_scaled, scaler = scale_features(df, fit=True, save_path=scaler_path)

    # Split stratifié
    X_train, X_test, y_train, y_test = split_data(
        df_scaled, test_size=test_size, random_state=random_state
    )

    # Rééquilibrage optionnel
    if use_smote:
        X_train, y_train = balance_data(X_train, y_train, random_state=random_state)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }