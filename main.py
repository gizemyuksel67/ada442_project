import streamlit as st
from Home import home_page
from Predictor import predictor_page
from Bank import bank_page

# Sidebar üzerinden sayfa seçimi
st.set_page_config(
    page_title="Streamlit Multi-Page App",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.sidebar.title("Menu")
page = st.sidebar.selectbox("Select a Page", ["Home", "Predictor", "Bank Analysis"])

# Sayfa geçişi
if page == "Home":
    home_page()
elif page == "Predictor":
    predictor_page()
elif page == "Bank Analysis":
    bank_page()
