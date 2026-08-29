import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Customer Segmentation Dashboard")

# Load dataset
df = pd.read_csv("customer_data.csv")

st.success("Customer dataset loaded successfully!")

# Basic information
st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    st.metric("Total Columns", len(df.columns))

with col3:
    st.metric("Missing Values", df.isnull().sum().sum())

st.subheader("📋 Customer Data")

st.dataframe(
    df,
    use_container_width=True
)
