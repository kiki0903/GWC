import streamlit as st
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# -----------------------------
# Page configuration & styling
# -----------------------------
st.set_page_config(
    page_title="Autism Traits Screener",
    page_icon="🧠",
    layout="wide"
)

# Small CSS tweaks for a cleaner, “app-like” look
st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1100px;
        margin: 0 auto;
    }
    .stMetric {
        background-color: #f8fafc;
        border-radius: 0.75rem;
        padding: 0.75rem 1rem;
        border: 1px solid #e5e7eb;
    }
    .card {
        padding: 1.25rem 1.5rem;
        border-radius: 1rem;
        border: 1px solid #e5e7eb;
        background-color: #ffffff;
        box-shadow: 0 8px 20px rgba(15,23,42,0.04);
    }
    .card-muted {
        padding: 1.25rem 1.5rem;
        border-radius: 1rem;
        border: 1px solid #e5e7eb;
        background-color: #f9fafb;
        box-shadow: 0 8px 20px rgba(15,23,42,0.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Load and prepare the dataset
# -----------------------------
@st.cache_data
def load_data():
    data = fetch_ucirepo(id=426)  # UCI Adult Autism dataset
    df = pd.concat([data.data.features, data.data.targets], axis=1)

    # Drop non-AQ / demo columns we are not using in the model
    df = df.drop(
        columns=[
            "country_of_res",
            "relation",
            "age_desc",
            "result",
            "age",
            "gender",
            "ethnicity",
            "used_app_before",
        ]
    )

    # Clean missing values
    df = df.replace("?", np.nan).dropna()

    # Encode features
    df["jaundice"] = df["jaundice"].map({"yes": 1, "no": 0})

    le_family = LabelEncoder()
    df["family_pdd"] = le_family.fit_transform(df["family_pdd"])

    # Target encoding
    df["class"] = df["class"].map({"NO": 0, "YES": 1})

    # Identify question columns (A1_Score … A10_Score)
    question_cols = [c for c in df.columns if c.startswith("A") and c.endswith("_Score")]
    question_cols = sorted(question_cols)  # A1…A10 in order

    return df, le_family, question_cols


@st.cache_resource
def train_model(df):
    X = df.drop(columns="class")
    y = df["class"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Dataset-level probability stats
    proba_all = model.predict_proba(X)[:, 1]
    stats = {
        "avg_all": float(np.mean(proba_all)),
        "avg_positive": float(np.mean(proba_all[y == 1])),
        "avg_negative": float(np.mean(proba_all[y == 0])),
        "p25": float(np.percentile(proba_all, 25)),
        "p50": float(np.percentile(proba_all, 50)),
        "p75": float(np.percentile(proba_all, 75)),
    }

    return model, X.columns.tolist(), stats


def main():
    # ---------------
    # Sidebar content
    # ---------------
    with st.sidebar:
        st.markdown("### ℹ️ About this tool")
        st.write(
            "This web app uses a machine-learning model trained on the "
            "UCI Adult Autism Screening dataset (AQ-10) to estimate how "
            "similar your answers are to people who screened positive for "
            "autism-related traits."
        )

        st.markdown("**Important:**")
        st.caption(
            "- This is **not** a diagnosis.\n"
            "- Results are for **education and self-reflection only**.\n"
            "- Talk to a qualified clinician if you have concerns."
        )

        st.markdown("---")
        st.caption("Data source: UCI Machine Learning Repository — Adult Autism Screening (ID: 426).")

    # ---------------
    # Hero / intro
    # ---------------
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("#### Autism Traits Screener")
        st.title("🧠 AQ-10–style machine-learning screening")
        st.write(
            "Answer ten short questions about how you experience social situations, "
            "communication, and attention. A trained Random Forest model will estimate the "
            "likelihood that your pattern of responses resembles those of adults who "
            "screened positive for autism-related traits in a research dataset."
        )

    with col_right:
        st.markdown("##### How this app works")
        st.write(
            "1. You answer 10 AQ-style questions.\n"
            "2. The model computes a probability score (0–100%).\n"
            "3. Your score is compared to averages in the dataset.\n"
        )

    st.markdown("---")

    # Load data & model
    df, le_family, question_cols = load_data()
    model, feature_cols, stats = train_model(df)

    # ----------------------
    # Question text mapping
    # ----------------------
    pretty_questions = [
        "I am highly sensitive to small sounds that others often miss.",
        "I tend to focus more on small details than the overall picture.",
        "I find it difficult to multitask or do more than one thing at a time.",
        "It’s hard for me to resume what I was doing after being interrupted.",
        "I struggle to understand implied meanings in conversations.",
        "I often can’t tell if someone is getting bored while I'm talking.",
        "I have difficulty figuring out characters’ intentions when reading stories.",
        "I enjoy collecting detailed information about specific categories of things (e.g., cars, birds).",
        "I find it hard to understand what someone is feeling just by looking at their face.",
        "I often find it difficult to understand what others intend to do.",
    ]

    assert len(question_cols) == len(pretty_questions), "Mismatch between AQ items and columns."

    # ---------------
    # Main form
    # ---------------
    st.markdown("### 📝 Screening questions")

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        with st.form("autism_form"):
            answers = []
            for label in pretty_questions:
                val = st.radio(
                    label,
                    ["Yes", "No"],
                    key=label,
                    horizontal=True,
                )
                answers.append(1 if val == "Yes" else 0)

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                jaundice = st.radio(
                    "Did you experience jaundice as a child (yellowing of the skin/eyes)?",
                    ["yes", "no"],
                    horizontal=True,
                )
            with col2:
                family_pdd = st.selectbox(
                    "Family history of autism or related developmental conditions?",
                    le_family.classes_,
                )

            submitted = st.form_submit_button("Run screening")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------
    # Handle result
    # ---------------
    if submitted:
        # Build input row as a dict keyed by feature names
        input_dict = {}
        for col, ans in zip(question_cols, answers):
            input_dict[col] = ans

        input_dict["jaundice"] = 1 if jaundice == "yes" else 0
        input_dict["family_pdd"] = le_family.transform([family_pdd])[0]

        # Convert to DataFrame in same order as training features
        input_df = pd.DataFrame([input_dict])[feature_cols]

        prediction = model.predict(input_df)[0]
        proba = float(model.predict_proba(input_df)[0][1])

        st.markdown("### 📊 Your screening result")
        st.markdown('<div class="card-muted">', unsafe_allow_html=True)

        # Top summary metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                label="Your estimated likelihood",
                value=f"{proba * 100:.1f}%",
            )
        with m2:
            st.metric(
                label="Average likelihood in dataset",
                value=f"{stats['avg_all'] * 100:.1f}%",
            )
        with m3:
            label_text = "Higher than dataset average" if proba > stats["avg_all"] else "Lower than dataset average"
            st.metric(label="Relative to dataset", value=label_text)

        # Text explanation
        if prediction == 1:
            st.warning(
                "### ⚠️ Model interpretation: *Likely positive pattern*\n"
                "Your answers are similar to people who **screened positive** for autism traits "
                "in the research dataset.\n\n"
                "This result **does not equal a diagnosis**. If this resonates with your lived "
                "experience, it may be worth discussing with a clinician or trusted professional."
            )
        else:
            st.success(
                "### ✅ Model interpretation: *No strong autism-like pattern*\n"
                "Your responses **do not strongly match** the pattern of people who screened "
                "positive for autism traits in the dataset.\n\n"
                "However, screening tools and machine-learning models are imperfect. "
                "If you feel that autism might still describe your experiences, it is completely "
                "valid to seek a professional opinion."
            )

        # --------------------
        # Comparison chart
        # --------------------
        st.markdown("#### How your score compares")

        labels = ["Your score", "Dataset average", "Average (positive group)", "Average (negative group)"]
        values = [
            proba,
            stats["avg_all"],
            stats["avg_positive"],
            stats["avg_negative"],
        ]

        fig, ax = plt.subplots(figsize=(7, 2.2))
        ax.barh(labels, values, height=0.4)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Estimated likelihood (0–1)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

        # Small note under chart
        st.caption(
            "The model’s averages are computed on the original research dataset. "
            "“Positive group” refers to participants labeled as having autism; "
            "“negative group” refers to participants labeled as not having autism."
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.caption(
            "This tool is for **educational and exploratory purposes only** and should not be "
            "used as the sole basis for any medical or life decisions."
        )


if __name__ == "__main__":
    main()
