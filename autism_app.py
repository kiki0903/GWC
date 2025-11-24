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

# Small CSS tweaks for a cleaner look
st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background-color: #f8fafc;
        border-radius: 0.75rem;
        padding: 0.75rem 1rem;
        border: 1px solid #e5e7eb;
    }
    .question-card {
        padding: 1.25rem 1.5rem;
        border-radius: 1rem;
        border: 1px solid #e5e7eb;
        background-color: #ffffff;
        box-shadow: 0 8px 20px rgba(15,23,42,0.03);
    }
    .result-card {
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

    # Keep only AQ-10 scores + jaundice + family_pdd + class
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
    return model, X.columns.tolist()


def main():
    # ---------------
    # Sidebar content
    # ---------------
    with st.sidebar:
        st.markdown("### ℹ️ About this tool")
        st.write(
            "This web app uses a machine-learning model trained on the "
            "UCI Adult Autism Screening dataset (AQ-10) to estimate the "
            "likelihood of autism-related traits based on your answers."
        )

        st.markdown("**Important:**")
        st.caption(
            "- This is **not** a diagnosis.\n"
            "- Results are for **education and self-reflection only**.\n"
            "- Please talk to a qualified clinician for any concerns."
        )

        st.markdown("---")
        st.caption("Data source: UCI Machine Learning Repository — Adult Autism Screening (ID: 426).")

    # ---------------
    # Hero / intro
    # ---------------
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("#### Autism Traits Screener")
        st.title("🧠 Quick AQ-10–style screening")
        st.write(
            "Answer a short set of questions about how you experience the world. "
            "A machine-learning model will estimate the likelihood that your "
            "responses resemble those of people who screened positive for "
            "autism-related traits in the research dataset."
        )

    with col_right:
        st.markdown("##### How it works")
        st.write(
            "1. You complete 10 trait questions.\n"
            "2. The model computes a probability score.\n"
            "3. You get a friendly explanation of the result."
        )

    st.markdown("---")

    # Load data & model
    df, le_family, question_cols = load_data()
    model, feature_cols = train_model(df)

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

    # Safety check: make sure there are 10 question columns
    assert len(question_cols) == len(pretty_questions), "Mismatch between AQ items and columns."

    # ---------------
    # Main form
    # ---------------
    st.markdown("### 📝 Screening questions")

    with st.container():
        st.markdown('<div class="question-card">', unsafe_allow_html=True)

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
                    "Family history of autism or other pervasive developmental conditions?",
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

        # Map AQ answers to their original A1_Score…A10_Score columns
        for col, ans in zip(question_cols, answers):
            input_dict[col] = ans

        input_dict["jaundice"] = 1 if jaundice == "yes" else 0
        input_dict["family_pdd"] = le_family.transform([family_pdd])[0]

        # Convert to DataFrame in the same column order as training
        input_df = pd.DataFrame([input_dict])[feature_cols]

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.markdown("### 📊 Your screening result")
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        # Metric row
        mcol1, mcol2 = st.columns([1, 1])
        with mcol1:
            st.metric(
                label="Estimated likelihood of autism-related traits",
                value=f"{proba * 100:.1f}%",
            )
        with mcol2:
            label_text = "Higher-than-average" if proba >= 0.5 else "Lower-than-average"
            st.metric(label="Pattern relative to dataset", value=label_text)

        # Text explanation
        if prediction == 1:
            st.warning(
                "### ⚠️ Model interpretation: *Likely positive pattern*\n"
                "Your answers are similar to people who screened **positive** for autism traits "
                "in the research dataset.\n\n"
                "**This is not a diagnosis.** A formal evaluation with a qualified clinician is "
                "the only way to diagnose autism. If you have concerns, consider sharing your "
                "experiences and this result with a healthcare professional."
            )
        else:
            st.success(
                "### ✅ Model interpretation: *No strong autism-like pattern*\n"
                "Your responses **do not strongly match** the pattern of people who screened "
                "positive for autism traits in the dataset.\n\n"
                "However, screening tools and machine-learning models are imperfect. "
                "If you still feel that autism might describe your experiences, it can still "
                "be worth talking with a clinician or trusted professional."
            )

        # Score bar
        st.markdown("#### Score position")
        fig, ax = plt.subplots(figsize=(6, 0.8))
        ax.barh(["Your score"], [proba], color="orange", height=0.3)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Estimated likelihood (0–1)")
        ax.set_yticks([0])
        ax.set_yticklabels(["Your score"], fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

        st.caption(
            "This tool is for **educational and exploratory purposes only** and should not be "
            "used as the sole basis for any medical or life decisions."
        )


if __name__ == "__main__":
    main()
