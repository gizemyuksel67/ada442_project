import streamlit as st
from streamlit_lottie import st_lottie
import json
import pandas as pd

def home_page():
    
    st.title("Deposit Decision Predictor")

    # Load data
    df = pd.read_csv("datasets/cleaned_data.csv")

    # Sidebar
    st.sidebar.header("Deposit Decision Predictor")
    st.sidebar.image("gadgets/home.jpeg")
    st.sidebar.markdown(
        "Made By : Basme Zantout, Zeynep Sude Bal, Gizem Yüksel, Ahmet Tokgöz"
    )

    # Animation upload and open
    with open("gadgets/animation.json") as source:
        animation = json.load(source)
    st_lottie(animation)

    # About Data section
    st.title("Bank Marketing Dataset Overview")

    # Dataset Overview
    st.header("Dataset Overview")
    st.write("""
    - The dataset comes from the UCI Machine Learning Repository and contains information on direct marketing campaigns by a Portuguese banking institution.
    """)

    # Main Objective
    st.header("Main Objective")
    st.write("""
    - Predict the outcome (yes or no) of whether a customer will subscribe to a term deposit.
    """)

    # Dataset Features
    st.header("Features Overview")
    st.subheader("Client Information")
    st.write("""
    - **age**, **job**, **marital**, **education**, **default**, **housing**, **loan**
    """)

    st.subheader("Contact Information")
    st.write("""
    - **contact**, **month**, **day_of_week**, **duration**
    """)

    st.subheader("Campaign Information")
    st.write("""
    - **campaign**, **pdays**, **previous**, **poutcome**
    """)

    st.subheader("Economic Context")
    st.write("""
    - **emp.var.rate**, **cons.price.idx**, **cons.conf.idx**, **euribor3m**, **nr.employed**
    """)

    st.header("Target Variable")
    st.write("**y**: Indicates if the client subscribed to a term deposit (yes/no).")

    btn = st.button("Show a random sample from the dataset")
    if btn:
        st.write(df.sample(5))
