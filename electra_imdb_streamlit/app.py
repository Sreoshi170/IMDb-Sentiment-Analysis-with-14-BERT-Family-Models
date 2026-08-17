"""Streamlit app for the fine-tuned ELECTRA IMDb sentiment model."""

from __future__ import annotations
import html
import json
import os
import re
from pathlib import Path
from typing import Iterable
import pandas as pd
import streamlit as st
import torch
from bs4 import BeautifulSoup
from transformers import AutoModelForSequenceClassification, AutoTokenizer


APP_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = APP_DIR / "model"
DEFAULT_MAX_LENGTH = 128
MAX_BATCH_ROWS = 5_000

FALLBACK_METRICS = {
    "accuracy": 0.905,
    "precision": 0.898238747553816,
    "recall": 0.9143426294820717,
    "f1": 0.9062191510365252,
    "roc_auc": 0.959027344437511,
}


st.set_page_config(
    page_title="IMDb Sentiment · ELECTRA",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 2rem;}
      .hero {
        padding: 1.7rem 1.9rem; border-radius: 18px; margin-bottom: 1rem;
        background: linear-gradient(120deg, #111827, #312e81 55%, #7c3aed);
        color: white; box-shadow: 0 12px 32px rgba(49,46,129,.24);
      }
      .hero h1 {margin: 0 0 .35rem 0; font-size: 2.15rem;}
      .hero p {margin: 0; opacity: .88; font-size: 1.03rem;}
      .result-positive, .result-negative {
        padding: 1.2rem 1.4rem; border-radius: 14px; margin: .5rem 0 1rem;
        color: white; font-size: 1.18rem; font-weight: 650;
      }
      .result-positive {background: linear-gradient(120deg, #047857, #10b981);}
      .result-negative {background: linear-gradient(120deg, #991b1b, #ef4444);}
      [data-testid="stMetric"] {
        background: rgba(127,127,127,.08); border: 1px solid rgba(127,127,127,.17);
        padding: .75rem; border-radius: 12px;
      }
      .small-note {opacity: .72; font-size: .88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def setting(name: str, default: str = "") -> str:
    """Read a Streamlit secret first, then an environment variable."""
    try:
        value = st.secrets.get(name, os.getenv(name, default))
    except (FileNotFoundError, KeyError):
        value = os.getenv(name, default)
    return str(value) if value is not None else default


def clean_review(text: object) -> str:
    """Apply the same lightweight cleanup used before model training."""
    value = html.unescape(str(text))
    value = BeautifulSoup(value, "html.parser").get_text(" ")
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def resolve_model_source() -> tuple[str, str | None, str]:
    """Prefer extracted local weights, otherwise use a Hugging Face repository."""
    if (LOCAL_MODEL_DIR / "model.safetensors").is_file():
        return str(LOCAL_MODEL_DIR), None, "Local model folder"

    repo_id = setting("HF_MODEL_REPO").strip()
    if not repo_id or repo_id.startswith("your-"):
        raise RuntimeError(
            "No model was found. Extract the supplied model ZIP into the model/ "
            "folder, or set HF_MODEL_REPO in Streamlit secrets."
        )
    token = setting("HF_TOKEN").strip() or None
    return repo_id, token, f"Hugging Face · {repo_id}"


@st.cache_resource(show_spinner="Loading ELECTRA and its tokenizer…")
def load_model(source: str, token: str | None):
    """Load the model once per Streamlit server instead of once per rerun."""
    tokenizer = AutoTokenizer.from_pretrained(source, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(source, token=token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def normalized_label(model, class_id: int) -> str:
    """Convert model label metadata into friendly POSITIVE/NEGATIVE names."""
    mapping = model.config.id2label or {}
    raw_label = str(mapping.get(class_id, mapping.get(str(class_id), class_id))).upper()
    if "POS" in raw_label or raw_label == "1":
        return "POSITIVE"
    if "NEG" in raw_label or raw_label == "0":
        return "NEGATIVE"
    return raw_label


def predict_reviews(
    reviews: Iterable[object], tokenizer, model, device, max_length: int, batch_size: int = 16
) -> pd.DataFrame:
    """Predict sentiment in small batches to control memory use."""
    original = [str(review) for review in reviews]
    cleaned = [clean_review(review) for review in original]
    rows: list[dict] = []

    with torch.inference_mode():
        for start in range(0, len(cleaned), batch_size):
            batch = cleaned[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            probabilities = torch.softmax(model(**encoded).logits, dim=-1).cpu()

            for offset, probability in enumerate(probabilities):
                class_id = int(torch.argmax(probability).item())
                label = normalized_label(model, class_id)
                negative_id = next(
                    (i for i in range(len(probability)) if normalized_label(model, i) == "NEGATIVE"),
                    0,
                )
                positive_id = next(
                    (i for i in range(len(probability)) if normalized_label(model, i) == "POSITIVE"),
                    min(1, len(probability) - 1),
                )
                rows.append(
                    {
                        "review": original[start + offset],
                        "cleaned_review": batch[offset],
                        "sentiment": label,
                        "confidence": float(probability[class_id]),
                        "negative_probability": float(probability[negative_id]),
                        "positive_probability": float(probability[positive_id]),
                    }
                )

    return pd.DataFrame(rows)


def load_metadata() -> dict:
    """Read training metadata when local files are available."""
    metadata_path = LOCAL_MODEL_DIR / "deployment_metadata.json"
    if metadata_path.is_file():
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"model_name": "ELECTRA-base", "max_length": DEFAULT_MAX_LENGTH, **FALLBACK_METRICS}


st.markdown(
    """
    <div class="hero">
      <h1>🎬 IMDb Review Sentiment</h1>
      <p>Fine-tuned ELECTRA-base · Predict whether a movie review is positive or negative.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metadata = load_metadata()
try:
    model_source, hf_token, source_description = resolve_model_source()
    tokenizer, model, device = load_model(model_source, hf_token)
except Exception as exc:
    st.error(f"The model could not be loaded: {exc}")
    st.info(
        "For local use, run `python scripts/setup_local_model.py --zip "
        "ELECTRA-base_IMDb_deployment.zip`. For Streamlit Cloud, set "
        "`HF_MODEL_REPO` in the app's Secrets. See README.md for the full walkthrough."
    )
    st.stop()

try:
    max_length = int(setting("MAX_LENGTH", str(metadata.get("max_length", DEFAULT_MAX_LENGTH))))
except ValueError:
    max_length = DEFAULT_MAX_LENGTH

with st.sidebar:
    st.header("Model status")
    st.success("Ready")
    st.caption(source_description)
    st.write(f"**Compute:** {str(device).upper()}")
    st.write(f"**Maximum tokens:** {max_length}")
    st.divider()
    st.subheader("Test-set performance")
    st.metric("Accuracy", f"{float(metadata.get('accuracy', FALLBACK_METRICS['accuracy'])):.1%}")
    st.metric("F1 score", f"{float(metadata.get('f1', FALLBACK_METRICS['f1'])):.1%}")
    st.metric("ROC AUC", f"{float(metadata.get('roc_auc', FALLBACK_METRICS['roc_auc'])):.1%}")
    st.caption("Measured on the notebook's held-out IMDb test split.")

single_tab, batch_tab, about_tab = st.tabs(["Single review", "Batch CSV", "About the model"])

with single_tab:
    st.subheader("Analyze one movie review")
    example_col1, example_col2, _ = st.columns([1, 1, 3])
    if example_col1.button("Positive example", use_container_width=True):
        st.session_state.review_text = (
            "A beautifully acted and emotionally satisfying film. The story stayed with me long after it ended."
        )
    if example_col2.button("Negative example", use_container_width=True):
        st.session_state.review_text = (
            "The plot was predictable, the dialogue felt unnatural, and the movie was much too long."
        )

    review_text = st.text_area(
        "Movie review",
        key="review_text",
        height=180,
        placeholder="Type or paste an IMDb-style movie review here…",
    )
    st.caption(f"{len(review_text):,} characters · long text is truncated to {max_length} tokens")

    if st.button("Analyze sentiment", type="primary", use_container_width=True):
        if not clean_review(review_text):
            st.warning("Please enter a review before running the prediction.")
        else:
            result = predict_reviews([review_text], tokenizer, model, device, max_length).iloc[0]
            sentiment = result["sentiment"]
            css_class = "result-positive" if sentiment == "POSITIVE" else "result-negative"
            icon = "👍" if sentiment == "POSITIVE" else "👎"
            st.markdown(
                f'<div class="{css_class}">{icon} {sentiment} · {result["confidence"]:.1%} confidence</div>',
                unsafe_allow_html=True,
            )

            score_frame = pd.DataFrame(
                {
                    "Sentiment": ["Negative", "Positive"],
                    "Probability": [result["negative_probability"], result["positive_probability"]],
                }
            ).set_index("Sentiment")
            chart_col, metric_col = st.columns([2, 1])
            chart_col.bar_chart(score_frame, horizontal=True, color="#7c3aed")
            metric_col.metric("Predicted class", sentiment.title())
            metric_col.metric("Confidence", f"{result['confidence']:.2%}")
            st.caption(
                "Confidence is the model's softmax probability, not a guarantee that the prediction is correct."
            )

with batch_tab:
    st.subheader("Analyze many reviews from a CSV file")
    st.write("Upload a CSV, choose the column containing review text, then download the predictions.")
    uploaded_file = st.file_uploader("CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"The CSV could not be read: {exc}")
        else:
            if input_df.empty or len(input_df.columns) == 0:
                st.warning("The uploaded CSV has no rows or columns.")
            else:
                default_index = list(input_df.columns).index("review") if "review" in input_df.columns else 0
                text_column = st.selectbox("Review text column", input_df.columns, index=default_index)
                st.dataframe(input_df.head(10), use_container_width=True)

                if len(input_df) > MAX_BATCH_ROWS:
                    st.warning(
                        f"This app processes at most {MAX_BATCH_ROWS:,} rows at once. "
                        "Only the first rows will be analyzed."
                    )

                if st.button("Run batch prediction", type="primary", use_container_width=True):
                    working_df = input_df.head(MAX_BATCH_ROWS).copy()
                    with st.spinner(f"Analyzing {len(working_df):,} reviews…"):
                        predictions = predict_reviews(
                            working_df[text_column].fillna(""), tokenizer, model, device, max_length
                        )
                    working_df["predicted_sentiment"] = predictions["sentiment"]
                    working_df["confidence"] = predictions["confidence"].round(6)
                    working_df["negative_probability"] = predictions["negative_probability"].round(6)
                    working_df["positive_probability"] = predictions["positive_probability"].round(6)

                    counts = working_df["predicted_sentiment"].value_counts().rename("Reviews")
                    result_col, chart_col = st.columns([2, 1])
                    result_col.dataframe(working_df, use_container_width=True)
                    chart_col.bar_chart(counts, color="#7c3aed")
                    st.download_button(
                        "Download predictions as CSV",
                        data=working_df.to_csv(index=False).encode("utf-8"),
                        file_name="electra_imdb_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

with about_tab:
    st.subheader("What this app does")
    st.write(
        "The app cleans the review, converts it into ELECTRA tokens, limits the input to "
        f"{max_length} tokens, and sends those tokens through the fine-tuned classification model. "
        "Softmax converts the two output logits into negative and positive probabilities."
    )
    st.subheader("Why ELECTRA performed well")
    st.write(
        "ELECTRA pre-trains its discriminator by detecting replaced tokens throughout every input "
        "sequence. Because it learns from every token rather than only a small masked subset, its "
        "pre-training is sample-efficient and produces strong language representations. IMDb is a "
        "large, balanced English binary-classification dataset, which is a particularly good match "
        "for ELECTRA-base. The final ranking can also depend on the chosen split, random seed, sequence "
        "length, learning rate, and number of epochs, so this result does not prove that ELECTRA is "
        "always the best model."
    )
    metric_cols = st.columns(5)
    for column, (name, key) in zip(
        metric_cols,
        [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"), ("ROC AUC", "roc_auc")],
    ):
        column.metric(name, f"{float(metadata.get(key, FALLBACK_METRICS[key])):.3f}")
    st.info(
        "This is a learning/demo application. Predictions can be wrong, especially for sarcasm, "
        "mixed opinions, very long reviews, or text unlike the IMDb training data."
    )

