import json
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="GenderLens",
    page_icon="GL",
    layout="wide"
)

# -----------------------------
# CUSTOM DARK THEME
# -----------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
}

.title-box {
    background: linear-gradient(135deg, #312E81, #4C1D95, #701A75);
    padding: 35px 40px;
    border-radius: 18px;
    margin-bottom: 30px;
    border: 1px solid #6D5ACF;
}

.title-box h1 {
    color: white;
    font-size: 42px;
    margin: 0;
}

.title-box p {
    color: #E9D5FF;
    font-size: 18px;
    margin-top: 10px;
}

[data-testid="stMetric"] {
    background-color: #171B26;
    border: 1px solid #373A50;
    padding: 20px;
    border-radius: 14px;
}

[data-testid="stMetricLabel"] {
    color: #A5B4FC !important;
}

[data-testid="stMetricValue"] {
    color: white !important;
}

h2 {
    color: #C4B5FD !important;
}

h3 {
    color: #DDD6FE !important;
}

.info-card {
    background-color: #171B26;
    border-left: 4px solid #8B5CF6;
    padding: 18px;
    border-radius: 10px;
    margin: 15px 0;
}

.method-card {
    background-color: #171B26;
    border: 1px solid #34384A;
    padding: 20px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# TITLE
# -----------------------------

st.markdown("""
<div class="title-box">
    <h1>GenderLens</h1>
    <p>
        Auditing and Mitigating Gender-Associated Patterns
        in Pretrained Word Embeddings
    </p>
</div>
""", unsafe_allow_html=True)

st.write(
    "GenderLens evaluates gender-associated patterns in pretrained "
    "Word2Vec embeddings using cosine similarity and evaluates the "
    "effect of gender-direction neutralization."
)


# -----------------------------
# LOAD RESULTS
# -----------------------------

with open("results/before_debiasing.json") as file:
    before_data = json.load(file)

with open("results/after_debiasing.json") as file:
    after_data = json.load(file)

before = pd.DataFrame(before_data["occupations"])
after = pd.DataFrame(after_data["occupations"])


# -----------------------------
# COMBINE RESULTS
# -----------------------------

comparison = before[
    ["word", "gender_association_score", "association_label"]
].merge(
    after[
        ["word", "gender_association_score", "association_label"]
    ],
    on="word",
    suffixes=("_before", "_after")
)


# -----------------------------
# METRICS
# -----------------------------

mean_before = comparison[
    "gender_association_score_before"
].abs().mean()

mean_after = comparison[
    "gender_association_score_after"
].abs().mean()

reduction = (
    (mean_before - mean_after) / mean_before
) * 100


# -----------------------------
# OVERVIEW
# -----------------------------

st.header("Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Occupations Analysed",
    len(comparison)
)

col2.metric(
    "Mean Absolute Association",
    f"{mean_after:.4f}"
)

col3.metric(
    "Overall Reduction",
    f"{reduction:.2f}%"
)


# -----------------------------
# BEFORE VS AFTER
# -----------------------------

st.header("Before vs After Gender Association")

chart_data = comparison.set_index("word")[
    [
        "gender_association_score_before",
        "gender_association_score_after"
    ]
]

chart_data.columns = ["Before", "After"]

st.bar_chart(chart_data)


# -----------------------------
# OCCUPATION RESULTS
# -----------------------------

st.header("Occupation-Level Results")

display_table = comparison.rename(
    columns={
        "word": "Occupation",
        "gender_association_score_before": "Before",
        "association_label_before": "Before Label",
        "gender_association_score_after": "After",
        "association_label_after": "After Label"
    }
)

st.dataframe(
    display_table,
    use_container_width=True,
    hide_index=True
)


# -----------------------------
# STATISTICAL VALIDATION
# -----------------------------

st.header("Statistical Validation")

col1, col2, col3 = st.columns(3)

col1.metric("Occupations", "22")
col2.metric("Permutation Tests", "10,000")
col3.metric("p-value", "0.0007")

st.markdown("""
<div class="info-card">
A one-sided paired sign-flip permutation test was used to evaluate
whether the magnitude of gender-association scores decreased after
gender-direction neutralization.
</div>
""", unsafe_allow_html=True)

st.info(
    "The p-value indicates that the observed reduction is unusual "
    "under the specified permutation model. It does not prove that "
    "the embedding is completely unbiased."
)


# -----------------------------
# SEMANTIC PRESERVATION
# -----------------------------

st.header("Semantic Preservation")

try:

    with open("results/semantic_preservation.json") as file:
        semantic_data = json.load(file)

    mean_change = semantic_data[
        "mean_absolute_similarity_change"
    ]

    st.metric(
        "Mean Absolute Similarity Change",
        f"{mean_change:.4f}"
    )

    st.markdown("""
    <div class="info-card">
    A manually curated set of occupation–semantic-anchor relationships
    was used to examine whether useful semantic relationships changed
    after gender-direction neutralization.
    </div>
    """, unsafe_allow_html=True)

except Exception:

    st.info(
        "Semantic preservation results are not available."
    )


# -----------------------------
# METHODOLOGY
# -----------------------------

st.header("Methodology")

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
    <div class="method-card">
    <h3>Gender Attribute Sets</h3>

    <b>Male attributes</b><br>
    he, him, man, boy

    <br><br>

    <b>Female attributes</b><br>
    she, her, woman, girl
    </div>
    """, unsafe_allow_html=True)


with col2:

    st.markdown("""
    <div class="method-card">
    <h3>Association Interpretation</h3>

    <b>Above 0.01</b> → Male-associated

    <br><br>

    <b>Below -0.01</b> → Female-associated

    <br><br>

    <b>Between -0.01 and 0.01</b> → Near-balanced
    </div>
    """, unsafe_allow_html=True)


st.subheader("Gender Association Score")

st.code(
    """
Mean cosine similarity with male attributes
-
Mean cosine similarity with female attributes
""",
    language="text"
)

st.caption(
    "These labels describe the direction of measured association. "
    "They do not imply that one gender is better or worse."
)


# -----------------------------
# PROJECT INFORMATION
# -----------------------------

st.header("Project Information")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Model", "Google News Word2Vec")
col2.metric("Dimensions", "300")
col3.metric("Method", "Cosine Similarity")
col4.metric("Mitigation", "Direction Neutralization")


# -----------------------------
# LIMITATIONS
# -----------------------------

st.header("Limitations")

st.markdown("""
- The analysis uses one pretrained Word2Vec model.
- Association does not establish discrimination or causation.
- Results depend on the selected gender attribute words.
- The occupation vocabulary is relatively small.
- Gender-direction neutralization does not guarantee removal of
  all gender-related information.
- Semantic preservation is evaluated using a small manually
  curated probe rather than a standardized benchmark.
""")
