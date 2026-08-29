# 💳 Détection de Fraude Bancaire

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.3-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/Licence-MIT-yellow.svg)](LICENSE)

Projet complet de **détection de fraude bancaire** basé sur le jeu de données
Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
Le dataset contient **284 807 transactions** anonymisées (colonnes `Time`,
`V1-V28`, `Amount`, `Class`) dont seulement **~0,17 %** sont des fraudes —
un cas d'école de **classification fortement déséquilibrée**.

---

## 📁 Structure du projet

```
bank-fraud-detection/
├── data/                        # À remplir avec creditcard.csv (Kaggle)
├── notebooks/
│   ├── 01_EDA.ipynb             # Analyse exploratoire des données
│   ├── 02_Preprocessing.ipynb   # Démo du pipeline de prétraitement
│   ├── 03_Modeling.ipynb        # Entraînement des 3 modèles
│   └── 04_Evaluation.ipynb      # Comparaison des modèles
├── src/
│   ├── preprocessing.py         # Chargement, normalisation, split, SMOTE
│   ├── train.py                 # Entraînement + sauvegarde (joblib)
│   └── predict.py               # Inférence sur une transaction
├── models/                      # Modèles entraînés (généré par train.py)
├── sql/
│   └── fraud_analysis.sql       # Requêtes d'analyse SQLite
├── app/
│   └── app.py                   # Interface Streamlit
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Cloner / copier le projet puis se placer dedans
cd bank-fraud-detection

# 2. Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Récupérer les données
#    Téléchargez "creditcard.csv" depuis Kaggle et placez-le dans data/
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

---

## 🚀 Utilisation

### 1. Analyse exploratoire (EDA)

```bash
jupyter notebook
# puis ouvrez notebooks/01_EDA.ipynb
```

### 2. Entraîner les modèles

```bash
python src/train.py                       # 3 modèles classiques
python src/train.py --smote               # idem + SMOTE sur l'entraînement
```

Résultats sauvegardés dans `models/` :
`logistic_regression.joblib`, `random_forest.joblib`, `xgboost.joblib`
(+ `scaler.joblib`).

### 3. Évaluer et comparer

Ouvrez `notebooks/04_Evaluation.ipynb` : métriques détaillées,
matrices de confusion, courbes **ROC-AUC** et surtout **PR-AUC**
(métrique prioritaire sur données déséquilibrées).

### 4. Prédiction en ligne de commande

```bash
python src/predict.py
```

ou via une API Python :

```python
from predict import FraudPredictor

predictor = FraudPredictor("models/xgboost.joblib")
result = predictor.predict({"Time": 0.0, "Amount": 149.62, "V1": -1.35, ...})
print(result)  # {'classe': 0, 'probabilite_fraude': 0.012, ...}
```

### 5. Interface web (Streamlit)

```bash
streamlit run app/app.py
```

Saisissez `Time`, `Amount` et les composantes `V1-V28`, puis obtenez un
verdict avec jauge de risque :

- 🟢 **Transaction normale**
- 🔴 **Risque de fraude : XX %**

---

## 🧠 Modèles & approche

| Modèle                | Gestion du déséquilibre      |
|-----------------------|------------------------------|
| Logistic Regression   | `class_weight="balanced"`    |
| Random Forest         | `class_weight="balanced"`    |
| XGBoost               | `scale_pos_weight` (auto)    |

Le prétraitement standardise `Time` et `Amount` (`StandardScaler`),
conserve l'ordre des features pour la reproductibilité et peut être
étendu avec **SMOTE** (`--smote`) pour un suréchantillonnage synthétique
de la classe minoritaire.

---

## 📊 Résultats

Résultats sur le **jeu de test** (56 962 transactions, seuil de décision 0.5) :

| Modèle               | ROC-AUC | PR-AUC | Précision | Rappel | F1 |
|----------------------|---------|--------|-----------|--------|----|
| XGBoost              | 0.9673  | **0.8852** | 0.89 | 0.85 | 0.87 |
| Random Forest        | **0.9828** | 0.8500 | 0.71 | 0.85 | 0.77 |
| Logistic Regression  | 0.9721  | 0.7189 | 0.06 | 0.92 | 0.11 |

> **XGBoost est retenu** : meilleure **PR-AUC** (0.8852), métrique
> prioritaire sur données déséquilibrées. Il est configuré comme modèle
> par défaut de `src/predict.py` et `app/app.py`.
> Les métriques Précision/Rappel/F1 concernent la classe **Fraude (1)**.

---

## 🧰 Technologies

- **Python** 3.9+
- **scikit-learn** : modèles classiques + preprocessing
- **XGBoost** : gradient boosting optimisé
- **imbalanced-learn** : SMOTE pour le déséquilibre
- **Pandas / NumPy** : manipulation des données
- **Matplotlib / Seaborn** : visualisations
- **Jupyter** : notebooks d'analyse
- **Streamlit** : interface utilisateur
- **SQLite** : requêtes d'analyse SQL
- **joblib** : sérialisation des modèles

---

## 📄 Licence

Projet à but pédagogique. Données : [Dataset Kaggle - Credit Card Fraud
Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(A. Dall'Amico et al., ULB).