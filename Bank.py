import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def bank_page():
    # Sayfa ayarları


    st.title("Bank Marketing Success Analysis")

    # Sidebar
    st.sidebar.header("Bank Marketing Success Analysis")
    st.sidebar.image("gadgets/data_analysis.jpeg")
    st.sidebar.markdown(
        "Made By : Basme Zantout, Zeynep Sude Bal, Gizem Yüksel, Ahmet Tokgöz"
    )

    # Veri yükleme
    try:
        df = pd.read_csv(r'datasets/bank-additional.csv', sep=';')
    except FileNotFoundError:
        st.error("The dataset file is missing. Please upload the file.")
        return

    # pdays sütununu 'contacted_before' olarak dönüştürme
    def is_contacted_before(x):
        return "no" if x == 999 else "yes"

    df["contacted_before"] = df["pdays"].apply(is_contacted_before)
    df.drop("pdays", axis=1, inplace=True)

    # Önemli Metrikler
    st.subheader("Important Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Current Campaign Successful Deposits",
        np.round(df[df["y"] == "yes"]["y"].count())
    )
    col2.metric(
        "Current Campaign Successful Deposits %",
        np.round(df[df["y"] == "yes"]["y"].count() / len(df) * 100, 2)
    )
    col3.metric(
        "Previous Campaign Successful Deposits",
        np.round(df[df["poutcome"] == "success"]["poutcome"].count())
    )
    col4.metric(
        "Previous Campaign Successful Deposits %",
        np.round(df[df["poutcome"] == "success"]["poutcome"].count() / len(df) * 100, 2)
    )

    # Histogram ve Grafik Fonksiyonları
    def plot_histogram(df, column):
        fig = px.histogram(
            df, x=column, nbins=20,
            title=f"Histogram of {column}",
            labels={column: column},
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    def plot_pie_chart(df, column):
        fig = px.pie(
            df, names=column,
            title=f"Pie Chart of {column}",
            hole=0.3,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    def plot_categorical_vs_target_histogram(df, categorical_column, target_column):
        fig = px.histogram(
            df, x=categorical_column, color=target_column,
            barmode="group",
            title=f"{categorical_column} vs Deposit Success (Yes/No)",
            text_auto=True,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Grafikler
    st.subheader("Histograms")
    c1, c2 = st.columns(2)
    with c1:
        plot_histogram(df, "age")
    with c2:
        plot_histogram(df, "campaign")

    st.subheader("Pie Charts")
    c1, c2 = st.columns(2)
    with c1:
        plot_pie_chart(df, "marital")
    with c2:
        plot_pie_chart(df, "contacted_before")

    st.subheader("Categorical vs Target")
    plot_categorical_vs_target_histogram(df, "job", "y")

    # Insights Bölümü
    st.header("Insights")
    insights = """
    1. **Job Role and Subscription Rate**:
        - Students and retired individuals are more likely to subscribe to a term deposit.
    2. **Marital Status Influence**:
        - Single clients have a higher subscription rate compared to married or divorced clients.
    3. **Education Level Impact**:
        - Higher education levels, such as high school, university, and personal courses, have higher subscription rates.
    4. **Housing Loan and Personal Loan Factors**:
        - Clients with housing loans are more likely to subscribe.
    5. **Contact Method**:
        - Contacting via cell phone yields better results.
    """
    st.markdown(insights)

