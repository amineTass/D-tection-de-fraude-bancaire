"""
Application Streamlit de détection de fraude bancaire.

Charge le meilleur modèle entraîné (src/train.py -> models/), propose
un formulaire de saisie pour une transaction (Time, Amount, V1-V28) et
affiche un verdict visuel avec jauge de risque.

Lancement :  streamlit run app/app.py
"""

import os
import sys

import pandas as pd
import streamlit as st

# Ajout du dossier src au chemin pour importer nos modules
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from predict import FraudPredictor, load_best_model

# ---------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Détection de fraude bancaire",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Détection de fraude bancaire")
st.caption(
    "Modèles entraînés sur le jeu de données Kaggle "
    "*Credit Card Fraud Detection* (0,17 % de fraudes)."
)

# ---------------------------------------------------------------------
# Chargement du modèle (en cache pour éviter de recharger à chaque clic)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement du meilleur modèle…")
def charger_modele():
    return load_best_model()


try:
    predictor = charger_modele()
except FileNotFoundError as e:
    st.error(str(e))
    st.info(
        "Lancez d'abord l'entraînement : `python src/train.py` "
        "(avec le fichier `data/creditcard.csv`)."
    )
    st.stop()

# ---------------------------------------------------------------------
# Formulaire de saisie
# ---------------------------------------------------------------------
st.sidebar.header("🧾 Transaction à analyser")

heure_transaction = st.sidebar.number_input(
    "Heure de la transaction (Time, secondes depuis la 1ère transaction)",
    min_value=0.0,
    value=0.0,
    step=100.0,
)

montant = st.sidebar.number_input(
    "Montant de la transaction (EUR)",
    min_value=0.0,
    value=100.0,
    step=10.0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Composantes V1 - V28")
st.sidebar.caption("Valeurs par défaut à 0 (ajustables si besoin)")

# Paramètres V1-V28 (défauts = 0, ajustables dans le menu dépliant)
with st.sidebar.expander("Champs avancés (V1-V28)", expanded=False):
    valeurs_v = {}
    for i in range(1, 29):
        valeurs_v[f"V{i}"] = st.number_input(
            f"V{i}", value=0.0, step=0.1, format="%.4f"
        )

# Buttons
analyser = st.sidebar.button("Analyser la transaction", type="primary")
reinitialiser = st.sidebar.button("Réinitialiser")

if reinitialiser:
    st.rerun()

transaction = {
    "Time": float(heure_transaction),
    "Amount": float(montant),
    **{k: float(v) for k, v in valeurs_v.items()},
}

# ---------------------------------------------------------------------
# Résultat
# ---------------------------------------------------------------------
st.header("Résultat de l'analyse")

if analyser:
    with st.spinner("Analyse en cours…"):
        resultat = predictor.predict(transaction)

    proba = resultat["probabilite_fraude"]
    classe = resultat["classe"]

    # Jauge visuelle de risque (0 -> 100 %)
    st.subheader("Jauge de risque de fraude")
    st.progress(int(proba * 100), text=f"Risque : {proba * 100:.2f} %")

    # Verdict coloré
    seuil = 0.5
    if classe == 1 and proba >= seuil:
        st.markdown(f"## 🔴 Risque de fraude : {proba * 100:.2f} %")
        st.error("Cette transaction semble frauduleuse. Blocage recommandé.")
    else:
        st.markdown(f"## 🟢 Transaction normale")
        st.success(
            f"Probabilité de fraude estimée : {proba * 100:.2f} % "
            "(sous le seuil de 50 %)."
        )

    # Détails techniques repliés
    with st.expander("Voir les détails techniques"):
        st.markdown(
            f"- **Modèle utilisé** : `{os.path.basename(predictor.model_path)}`"
        )
        st.markdown(f"- **Classe prédite** : {classe}")
        st.markdown(f"- **Probabilité de fraude** : {proba:.6f}")
        st.markdown(f"- **Seuil de décision** : {seuil}")
        st.dataframe(pd.DataFrame(transaction, index=[0]))

else:
    st.info(
        "👈 Renseignez les caractéristiques de la transaction dans le "
        "panneau de gauche puis cliquez sur **Analyser la transaction**."
    )

# ---------------------------------------------------------------------
# Pied de page
# ---------------------------------------------------------------------
st.divider()
st.caption(
    "⚠️ Outil de démonstration — ne pas utiliser pour une décision "
    "d'anti-fraude réelle sans validation métier."
)