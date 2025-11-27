from pathlib import Path
import re
import pickle
import joblib
import numpy as np

import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ---------------------------------------------------------
# Basic config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cross-Platform Mental Health Text Lab",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"


# ---------------------------------------------------------
# Helpers: labels, text cleaning, etc.
# ---------------------------------------------------------
def clean_text_tfidf(text: str) -> str:
    """Match TF–IDF preprocessing (URLs, non-letters, whitespace, lowercase)."""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def label_to_text(label_int: int) -> str:
    """Binary label -> to friendly string."""
    return "Mental-health-related" if int(label_int) == 1 else "Control / non-mental-health"


def label_to_color(label_int: int) -> str:
    return "#f97373" if int(label_int) == 1 else "#69c36b"


# ---------------------------------------------------------
# Load models (cached) – MATCHING YOUR FOLDER STRUCTURE
# ---------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_tfidf_models():
    """
    Load TF–IDF-based models using:

    models/
      logreg/
        logreg_model.pkl
        tfidf_reddit_vectorizer.pkl
      svm/
        svm_model.pkl
        tfidf_vectorizer.pkl
      rf/
        rf_model.pkl
        tfidf_rf.pkl
    """
    models = {}

    logreg_dir = MODELS_DIR / "logreg"
    svm_dir = MODELS_DIR / "svm"
    rf_dir = MODELS_DIR / "rf"

    # --- Logistic Regression ---
    logreg_model_path = logreg_dir / "logreg_model.pkl"
    logreg_vec_path = logreg_dir / "tfidf_reddit_vectorizer.pkl"
    if logreg_model_path.exists() and logreg_vec_path.exists():
        models["logreg"] = {
            "name": "Logistic Regression (TF–IDF)",
            "clf": joblib.load(logreg_model_path),
            "vec": joblib.load(logreg_vec_path),
        }

    # --- SVM ---
    svm_model_path = svm_dir / "svm_model.pkl"
    svm_vec_path = svm_dir / "tfidf_vectorizer.pkl"
    if svm_model_path.exists() and svm_vec_path.exists():
        models["svm"] = {
            "name": "Support Vector Machine (TF–IDF)",
            "clf": joblib.load(svm_model_path),
            "vec": joblib.load(svm_vec_path),
        }

    # --- Random Forest ---
    rf_model_path = rf_dir / "rf_model.pkl"
    rf_vec_path = rf_dir / "tfidf_rf.pkl"
    if rf_model_path.exists() and rf_vec_path.exists():
        models["rf"] = {
            "name": "Random Forest (TF–IDF, domain-mixed)",
            "clf": joblib.load(rf_model_path),
            "vec": joblib.load(rf_vec_path),
        }

    return models


@st.cache_resource(show_spinner=True)
def load_bilstm_model():
    """
    Load BiLSTM model + tokenizer using your structure:

    models/
      bilstm/
        bilstm_model.h5
        tokenizer.pkl
        bilstm_config.json
    """
    bilstm_dir = MODELS_DIR / "bilstm"
    model_path = bilstm_dir / "bilstm_model.h5"
    tok_path = bilstm_dir / "tokenizer.pkl"

    if not (model_path.exists() and tok_path.exists()):
        return None

    model = load_model(model_path)
    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)

    return {
        "name": "BiLSTM (GloVe embeddings)",
        "model": model,
        "tokenizer": tokenizer,
        "max_len": 128,  # same as in training
    }


# ---------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------
def predict_with_tfidf(text: str, clf, vec):
    cleaned = clean_text_tfidf(text)
    X = vec.transform([cleaned])
    # If the classifier supports predict_proba:
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0][1]
    else:
        # Fallback: use decision_function scaled into (0,1) with a sigmoid-like map
        if hasattr(clf, "decision_function"):
            score = clf.decision_function(X)[0]
            proba = 1 / (1 + np.exp(-score))
        else:
            # Final fallback: treat predicted class as 0/1 with fixed confidence
            pred = clf.predict(X)[0]
            proba = 0.9 if int(pred) == 1 else 0.1
    pred_label = int(clf.predict(X)[0])
    return pred_label, float(proba)


def predict_with_bilstm(text: str, config: dict):
    model = config["model"]
    tokenizer = config["tokenizer"]
    max_len = config["max_len"]

    cleaned = clean_text_tfidf(text)  # reuse same basic cleaning
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    proba = float(model.predict(padded, verbose=0)[0][0])
    pred_label = 1 if proba >= 0.5 else 0
    return pred_label, proba


# ---------------------------------------------------------
# Plot helper: evaluation curves from output/
# ---------------------------------------------------------
def show_model_curves(model_key: str, train_domain: str, test_domain: str):
    """
    model_key: 'logreg', 'svm', 'rf', 'bilstm', 'bert'
    train_domain: 'reddit'
    test_domain: 'reddit' or 'twitter'
    Filenames like I named them: logreg_CM_reddit_reddit.png, logreg_ROC_reddit_twitter.png, etc.
    """
    prefix = model_key.lower()
    train_tag = "reddit"
    test_tag = test_domain.lower()

    cm_path = OUTPUT_DIR / f"{prefix}_CM_{train_tag}_{test_tag}.png"
    roc_path = OUTPUT_DIR / f"{prefix}_ROC_{train_tag}_{test_tag}.png"
    pr_path = OUTPUT_DIR / f"{prefix}_PR_{train_tag}_{test_tag}.png"

    any_plot = False

    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion Matrix", use_container_width=False)
        any_plot = True
    if roc_path.exists():
        st.image(str(roc_path), caption="ROC Curve", use_container_width=False)
        any_plot = True
    if pr_path.exists():
        st.image(str(pr_path), caption="Precision–Recall Curve", use_container_width=False)
        any_plot = True

    if not any_plot:
        st.info("No plots found for this configuration in the `output/` folder.")


# ---------------------------------------------------------
# Accuracy table from your report (Table VIII)
# All values are percentages.
# ---------------------------------------------------------
def get_accuracy_table():
    import pandas as pd

    data = [
        ["Logistic Regression", 94.0, 55.0, 70.0, 94.0, 60.0],
        ["Support Vector Machine", 93.0, 79.72, 91.96, 92.42, 54.72],
        ["Random Forest", 98.03, 87.07, 90.97, 86.73, 87.07],
        ["BiLSTM", 95.0, 56.0, 86.0, 96.0, 66.0],
        ["BERT", 96.95, 56.0, 87.0, 96.95, 86.95],
    ]
    cols = ["Model", "R→R", "R→T", "T→T", "Mix→R", "Mix→T"]
    df = pd.DataFrame(data, columns=cols)
    return df


# ---------------------------------------------------------
# Small helper to render one model's prediction nicely
# ---------------------------------------------------------
def display_prediction_card(model_name, label, prob, threshold, note, assumed_domain):
    label_text = label_to_text(label)
    color = label_to_color(label)
    prob_pct = prob * 100

    st.markdown(
        f"""
<div style="
    border: 1px solid #e0e0e0;
    border-radius: 0.6rem;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
">
  <div style="font-weight: 600; margin-bottom: 0.25rem;">{model_name}</div>
  <div style="display: flex; align-items: center; gap: 0.5rem;">
    <span style="
        background-color: {color};
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.9rem;
    ">{label_text}</span>
    <span style="font-size: 0.9rem;">
        Confidence: <strong>{prob_pct:.1f}%</strong>
        (threshold = {threshold:.2f})
    </span>
  </div>
  <div style="font-size: 0.8rem; color: #666; margin-top: 0.3rem;">
    Input treated as: <em>{assumed_domain}</em> · {note}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
def main():
    tfidf_models = load_tfidf_models()
    bilstm_cfg = load_bilstm_model()

    st.markdown(
        "<h1 style='text-align: center;'>🧠 Cross-Platform Mental Health Text Lab</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; margin-bottom: 1.5rem;'>"
        "Interactive demo for binary mental-health detection on Reddit & Twitter text, "
        "based on five models studied in COEN 498.</p>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["🔍 Interactive Demo", "📊 Model Results", "📚 Project Info"])

    # ------------------------------------------------------------------
    # TAB 1: Interactive Demo
    # ------------------------------------------------------------------
    with tabs[0]:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("Try a Reddit post or tweet")
            user_text = st.text_area(
                "Input text",
                height=180,
                placeholder="Paste or type a post here...",
            )

        with col_right:
            st.subheader("Settings")

            available_models = []
            if "logreg" in tfidf_models:
                available_models.append("Logistic Regression")
            if "svm" in tfidf_models:
                available_models.append("SVM")
            if "rf" in tfidf_models:
                available_models.append("Random Forest")
            if bilstm_cfg is not None:
                available_models.append("BiLSTM")

            model_choice = st.multiselect(
                "Models to run",
                options=available_models,
                default=available_models,  # run all working models by default
            )

            assumed_domain = st.selectbox(
                "Assumed platform of this text",
                options=["Reddit-style", "Twitter-style", "Unknown"],
                index=0,
            )

            threshold = st.slider(
                "Display as mental-health if probability ≥",
                min_value=0.5,
                max_value=0.9,
                value=0.5,
                step=0.05,
            )

            run_button = st.button("Analyze Text 🚀")

        st.markdown("---")

        if run_button:
            if not user_text.strip():
                st.warning("Please enter some text first.")
            elif not model_choice:
                st.warning("Please select at least one model.")
            else:
                st.subheader("Model predictions")

                for chosen in model_choice:
                    if chosen == "Logistic Regression" and "logreg" in tfidf_models:
                        cfg = tfidf_models["logreg"]
                        try:
                            label, prob = predict_with_tfidf(
                                user_text, cfg["clf"], cfg["vec"]
                            )
                            display_prediction_card(
                                model_name=cfg["name"],
                                label=label,
                                prob=prob,
                                threshold=threshold,
                                note="TF–IDF linear baseline",
                                assumed_domain=assumed_domain,
                            )
                        except Exception as e:
                            st.error(f"Logistic Regression error: {e}")

                    elif chosen == "SVM" and "svm" in tfidf_models:
                        cfg = tfidf_models["svm"]
                        try:
                            label, prob = predict_with_tfidf(
                                user_text, cfg["clf"], cfg["vec"]
                            )
                            display_prediction_card(
                                model_name=cfg["name"],
                                label=label,
                                prob=prob,
                                threshold=threshold,
                                note="TF–IDF margin-based classifier",
                                assumed_domain=assumed_domain,
                            )
                        except Exception as e:
                            st.error(f"SVM error: {e}")

                    elif chosen == "Random Forest" and "rf" in tfidf_models:
                        cfg = tfidf_models["rf"]
                        try:
                            label, prob = predict_with_tfidf(
                                user_text, cfg["clf"], cfg["vec"]
                            )
                            display_prediction_card(
                                model_name=cfg["name"],
                                label=label,
                                prob=prob,
                                threshold=threshold,
                                note="Trained with 80% Reddit + 20% Twitter",
                                assumed_domain=assumed_domain,
                            )
                        except Exception as e:
                            st.error(f"Random Forest error: {e}")

                    elif chosen == "BiLSTM" and bilstm_cfg is not None:
                        try:
                            label, prob = predict_with_bilstm(user_text, bilstm_cfg)
                            display_prediction_card(
                                model_name=bilstm_cfg["name"],
                                label=label,
                                prob=prob,
                                threshold=threshold,
                                note="Sequence model with pretrained embeddings",
                                assumed_domain=assumed_domain,
                            )
                        except Exception as e:
                            st.error(f"BiLSTM error: {e}")

                st.caption(
                    "Note: Fine-tuned BERT is evaluated in the results tab but "
                    "its checkpoint cannot be safely loaded for live predictions "
                    "due to a state-dict shape mismatch."
                )

    # ------------------------------------------------------------------
    # TAB 2: Model Results (accuracies + plots)
    # ------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Cross-platform accuracy summary")

        acc_df = get_accuracy_table()
        st.dataframe(acc_df, width=1200)

        st.markdown("---")
        st.subheader("Evaluation curves from saved experiments")

        col1, col2 = st.columns(2)
        with col1:
            model_for_plots = st.selectbox(
                "Select model",
                options=["Logistic Regression", "SVM", "Random Forest", "BiLSTM", "BERT"],
                index=2,
            )
        with col2:
            test_domain = st.selectbox(
                "Test domain for plots",
                options=["Reddit", "Twitter"],
                index=1,
            )

        key_map = {
            "Logistic Regression": "logreg",
            "SVM": "svm",
            "Random Forest": "rf",
            "BiLSTM": "bilstm",
            "BERT": "bert",
        }
        model_key = key_map[model_for_plots]

        if model_key == "bert":
            st.info(
                "BERT curves are loaded only from PNGs under `output/`. "
                "If the files are present, they will be shown below."
            )

        show_model_curves(model_key=model_key, train_domain="reddit", test_domain=test_domain.lower())

    # ------------------------------------------------------------------
    # TAB 3: Project Info
    # ------------------------------------------------------------------
    with tabs[2]:
        st.subheader("Project overview")

        st.markdown(
            """
This tool is part of a COEN 498 project on **cross-platform mental-health detection**.

- **Goal:** Study how well text classifiers trained on one platform (Reddit) generalize to another (Twitter).
- **Platforms:** Reddit (long, narrative posts) and Twitter (short, conversational tweets).
- **Task:** Binary classification  
  - **1 → Mental-health-related**  
  - **0 → Control / non-mental-health**
- **Models compared:** Logistic Regression, SVM, Random Forest, BiLSTM, and fine-tuned BERT.
- **Key finding:** All models perform well *within* a platform, but cross-platform accuracy can drop by up to ~40%.  
  Mixing just **20% Twitter data** into Reddit training dramatically improves BERT and Random Forest performance.
"""
        )

        st.markdown("---")
        st.subheader("Datasets")

        st.markdown(
            """
- **Reddit:** 28 subreddits total; 12 representative used in analysis  
  - Mental-health: *Addiction, ADHD, Alcoholism, Anxiety, MentalHealth, SuicideWatch*  
  - Control: *Conspiracy, Fitness, Jokes, Meditation, Relationship(s), Teaching*
- **Twitter:** 20,000 labeled English tweets (10k mental-health, 10k control).

Preprocessing follows the pipeline used in the report: URL removal, light cleaning, TF–IDF or embeddings, and 70/15/15 splits with additional 80% Reddit + 20% Twitter experiments for domain adaptation.
"""
        )

        st.markdown("---")
        st.subheader("Team")

        st.markdown(
            """
- **Hala Belamri** — Electrical and Computer Engineering, Concordia University, Montréal, Canada - ID: 40221036  
- **Minhtu Chau** — Electrical and Computer Engineering, Concordia University, Montréal, Canada - ID: 40232883  
- **Suomi Seki** — Electrical and Computer Engineering, Concordia University, Montréal, Canada - ID: 40328157  
- **James Johnson** — Electrical and Computer Engineering, Concordia University, Montréal, Canada - ID: 40253492  
- **Arefe Emrani** — Electrical and Computer Engineering, Concordia University, Montréal, Canada - ID: 40259157
"""
        )

        st.caption(
            "This demo focuses on the *binary* task (mental-health vs control). "
            "More fine-grained labels like 'depression' or 'PTSD' would require "
            "a different dataset and multi-class training setup."
        )


if __name__ == "__main__":
    main()
