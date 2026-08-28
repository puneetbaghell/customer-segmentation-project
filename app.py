import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="👥",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("👥 Customer Segmentation Dashboard")
st.markdown(
    "### Customer analysis using **K-Means Clustering**"
)

st.divider()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

try:
    df = pd.read_csv("customer_data.csv")
except Exception as e:
    st.error("❌ customer_data.csv file could not be loaded.")
    st.write(e)
    st.stop()

# Clean column names
df.columns = df.columns.str.strip()

# --------------------------------------------------
# FIND IMPORTANT COLUMNS
# --------------------------------------------------

def find_column(possible_names):
    for name in possible_names:
        for col in df.columns:
            if col.lower().replace("_", " ").strip() == name.lower():
                return col
    return None


income_col = find_column([
    "Annual Income",
    "Annual Income (k$)",
    "Income",
    "Income (k$)"
])

spending_col = find_column([
    "Spending Score",
    "Spending Score (1-100)",
    "Spending"
])

age_col = find_column([
    "Age"
])

gender_col = find_column([
    "Gender",
    "Sex"
])

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("⚙️ Dashboard Settings")

# Number of clusters
k = st.sidebar.slider(
    "Number of Customer Segments",
    min_value=2,
    max_value=10,
    value=5
)

# --------------------------------------------------
# CREATE CLUSTERS
# --------------------------------------------------

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# Remove ID-like columns
feature_cols = [
    col for col in numeric_cols
    if not any(x in col.lower() for x in ["id", "customerid"])
]

# Prefer income + spending score
if income_col and spending_col:
    feature_cols = [income_col, spending_col]

if len(feature_cols) < 2:
    st.error(
        "❌ At least two numerical columns are required for clustering."
    )
    st.write("Available numerical columns:", numeric_cols)
    st.stop()

# Fill missing values
model_data = df[feature_cols].copy()

for col in feature_cols:
    model_data[col] = pd.to_numeric(
        model_data[col],
        errors="coerce"
    )

model_data = model_data.fillna(model_data.median())

# Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(model_data)

# K-Means
kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["Segment"] = kmeans.fit_predict(scaled_data) + 1

# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "👥 Total Customers",
        len(df)
    )

with col2:
    st.metric(
        "🎯 Number of Segments",
        df["Segment"].nunique()
    )

if income_col:
    with col3:
        st.metric(
            "💰 Avg Income",
            f"{df[income_col].mean():.2f}"
        )
else:
    with col3:
        st.metric(
            "📊 Numeric Features",
            len(feature_cols)
        )

if spending_col:
    with col4:
        st.metric(
            "⭐ Avg Spending Score",
            f"{df[spending_col].mean():.2f}"
        )
else:
    with col4:
        st.metric(
            "📈 Features Used",
            len(feature_cols)
        )

st.divider()

# --------------------------------------------------
# SEGMENT DISTRIBUTION
# --------------------------------------------------

st.subheader("📊 Customer Segment Distribution")

segment_counts = df["Segment"].value_counts().sort_index()

col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        segment_counts.index.astype(str),
        segment_counts.values
    )

    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customers per Segment")

    st.pyplot(fig)

with col2:

    st.dataframe(
        segment_counts.rename(
            "Number of Customers"
        ).to_frame(),
        use_container_width=True
    )

# --------------------------------------------------
# INCOME VS SPENDING
# --------------------------------------------------

if income_col and spending_col:

    st.subheader("💰 Income vs Spending Score")

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        df[income_col],
        df[spending_col],
        c=df["Segment"],
        cmap="viridis",
        s=70,
        alpha=0.8
    )

    ax.set_xlabel(income_col)
    ax.set_ylabel(spending_col)

    ax.set_title(
        "Customer Segments: Income vs Spending"
    )

    legend = ax.legend(
        *scatter.legend_elements(),
        title="Segment"
    )

    ax.add_artist(legend)

    st.pyplot(fig)

# --------------------------------------------------
# SEGMENT SUMMARY
# --------------------------------------------------

st.subheader("📋 Segment Summary")

summary = df.groupby("Segment").agg(
    Customers=("Segment", "count")
)

if age_col:
    summary["Average Age"] = df.groupby(
        "Segment"
    )[age_col].mean().round(2)

if income_col:
    summary["Average Income"] = df.groupby(
        "Segment"
    )[income_col].mean().round(2)

if spending_col:
    summary["Average Spending Score"] = df.groupby(
        "Segment"
    )[spending_col].mean().round(2)

st.dataframe(
    summary,
    use_container_width=True
)

# --------------------------------------------------
# CUSTOMER DATA
# --------------------------------------------------

st.subheader("🔎 Customer Data")

# Segment filter
selected_segments = st.multiselect(
    "Select Customer Segments",
    sorted(df["Segment"].unique()),
    default=sorted(df["Segment"].unique())
)

filtered_df = df[
    df["Segment"].isin(selected_segments)
]

st.dataframe(
    filtered_df,
    use_container_width=True,
    height=400
)

# --------------------------------------------------
# DOWNLOAD
# --------------------------------------------------

st.subheader("⬇️ Download Segmented Data")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Customer Segments CSV",
    data=csv,
    file_name="customer_segments_dashboard.csv",
    mime="text/csv"
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    **Customer Segmentation Project**

    Developed using Python, Pandas, Scikit-learn,
    Matplotlib and Streamlit.

    **Algorithm:** K-Means Clustering
    """
)