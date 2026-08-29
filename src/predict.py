"""
Module de prédiction pour la détection de fraude bancaire.

Charge un modèle entraîné (sauvegardé par train.py) et prédit le risque
de fraude d'une nouvelle transaction, en acceptant un dictionnaire ou un
DataFrame en entrée. Reproduit exactement le prétraitement appliqué à
l'entraînement (StandardScaler sur Time et Amount + ordre des features).
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from preprocessing import FEATURE_COLUMNS, TARGET_COLUMN

MODELS_DIR = "models"
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")


class FraudPredictor:
    """
    Classe d'inférence : encapsule un modèle + le scaler pour prédire
    le risque de fraude d'une transaction.

    Paramètres
    ----------
    model_path : str
        Chemin vers le fichier joblib généré par train.py.
    scaler_path : str, optionnel
        Chemin vers le scaler sauvegardé pendant l'entraînement.
    """

    def __init__(self, model_path, scaler_path=SCALER_PATH):
        self.model_path = model_path
        bundle = joblib.load(model_path)

        self.model = bundle["model"]
        self.features = bundle.get("features")
        self.target = bundle.get("target", TARGET_COLUMN)

        # Vérification de la présence du scaler (sinon on cherche à l'entraîner)
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            print(
                f"Scaler non trouvé ({scaler_path}) : la normalisation "
                "sera basée sur celle du modèle (attendu)."
            )
            self.scaler = StandardScaler()

    def format_input(self, transaction):
        """
        Transforme l'entrée (dict ou DataFrame) en DataFrame avec les
        features dans l'ordre attendu par le modèle.

        Paramètres
        ----------
        transaction : dict | pd.DataFrame
            Transaction à prédire (Time, Amount, V1-V28).

        Retourne
        --------
        pd.DataFrame
        """
        if isinstance(transaction, dict):
            df = pd.DataFrame([transaction])
        elif isinstance(transaction, pd.DataFrame):
            df = transaction.copy()
        else:
            raise TypeError(
                "L'entrée doit être un dict ou un DataFrame pandas, "
                f"reçu : {type(transaction)}."
            )

        # On garantit l'ordre et la présence des features du modèle
        required = self.features or (FEATURE_COLUMNS + ["Time", "Amount"])
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                f"Colonnes manquantes dans la transaction : {missing}."
            )
        return df[required]

    def predict(self, transaction, threshold=0.5):
        """
        Prédit la classe et la probabilité de fraude.

        Paramètres
        ----------
        transaction : dict | pd.DataFrame
            Transaction à évaluer.
        threshold : float
            Seuil de décision pour la classe (défaut 0.5).

        Retourne
        --------
        dict
            {"classe": int, "probabilite_fraude": float, "label": str}
        """
        df = self.format_input(transaction)

        # Application du même prétraitement qu'à l'entraînement
        df_scaled = df.copy()
        df_scaled[["Time", "Amount"]] = self.scaler.transform(
            df_scaled[["Time", "Amount"]]
        )

        # Pour un modèle scikit-learn / xgboost, on prédit la proba
        proba_fraude = float(self.model.predict_proba(df_scaled)[:, 1][0])
        classe = int(proba_fraude >= threshold)

        return {
            "classe": classe,
            "probabilite_fraude": proba_fraude,
            "label": "FRAUDE" if classe == 1 else "NORMALE",
        }


def load_best_model(models_dir=MODELS_DIR):
    """
    Charge le meilleur modèle disponible dans le dossier models/.
    Priorité : xgboost > random_forest > logistic_regression.

    Retourne
    --------
    FraudPredictor
    """
    prioritie = ["xgboost", "random_forest", "logistic_regression"]
    for name in prioritie:
        path = os.path.join(models_dir, f"{name}.joblib")
        if os.path.exists(path):
            print(f"Modèle chargé : {path}")
            return FraudPredictor(path)
    raise FileNotFoundError(
        f"Aucun modèle trouvé dans {models_dir}. "
        "Lancez d'abord : python src/train.py"
    )


if __name__ == "__main__":
    # Exemple d'utilisation en ligne de commande
    predictor = load_best_model()

    exemple = {
        "Time": 0.0,
        "V1": -1.359807, "V2": -0.072781, "V3": 2.536346, "V4": 1.378155,
        "V5": -0.338320, "V6": 0.462387, "V7": 0.239598, "V8": 0.098698,
        "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
        "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470400,
        "V17": 0.207971, "V18": 0.025790, "V19": 0.403993, "V20": 0.251412,
        "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
        "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
        "Amount": 149.62,
    }

    resultat = predictor.predict(exemple)
    print(f"Classe : {resultat['classe']} ({resultat['label']})")
    print(f"Probabilité de fraude : {resultat['probabilite_fraude']:.4f}")