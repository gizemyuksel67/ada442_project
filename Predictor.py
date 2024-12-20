import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import pickle as pkl

def predictor_page():
    # Streamlit sayfa ayarları
    st.title("Term Deposit Success Predictor")

    # Sidebar
    st.sidebar.header("Term Deposit Success Predictor")
    st.sidebar.image("gadgets/data_analysis.jpeg")
    st.sidebar.subheader("Choose Your Favorite Predictor")

    # Model seçimi için bir selectbox
    model_choice = st.sidebar.selectbox(
        "Predictors", ["Random Forest", "Logistic Regression", "KNN"]
    )

    # Veri yükleme
    df = pd.read_csv("datasets/cleaned_data_v2.csv")

    # Kategorik ve sayısal sütunları tanımlayın
    categorical_columns = [
        'JobType', 'MaritalStatus', 'EducationLevel', 'HasHousingLoan',
        'HasPersonalLoan', 'ContactCommunicationType', 'LastContactMonth',
        'LastContactDayOfWeek', 'PreviousCampaignOutcome'
    ]
    numerical_columns = [
        'Age', 'CallDuration', 'CampaignContacts', 'PreviousCampaignContacts',
        'EmploymentVariationRate', 'ConsumerPriceIndex', 'ConsumerConfidenceIndex',
        'Euribor3MRate', 'NumberOfEmployees'
    ]

    # Önceden eğitilmiş modeller ve ön işleyiciyi yükleme
    rf = pkl.load(open('models/best_rf2_model.pkl', 'rb'))
    logreg = pkl.load(open('models/best_logreg2_model.pkl', 'rb'))
    knn = pkl.load(open('models/best_knn2_model.pkl', 'rb'))
    preprocessor = pkl.load(open('models/preprocessor2.pkl', 'rb'))

    # Seçilen model
    if model_choice == "Random Forest":
        model = rf
    elif model_choice == "Logistic Regression":
        model = logreg
    elif model_choice == "KNN":
        model = knn

    # Kullanıcıdan giriş verilerini alma
    col1, col2 = st.columns(2)

    # Sol kolon girişleri
    with col1:
        job_type = st.selectbox('Job Type', df['JobType'].unique())
        marital_status = st.selectbox('Marital Status', df['MaritalStatus'].unique())
        education_level = st.selectbox('Education Level', df['EducationLevel'].unique())
        has_housing_loan = st.selectbox('Has Housing Loan', df['HasHousingLoan'].unique())
        has_personal_loan = st.selectbox('Has Personal Loan', df['HasPersonalLoan'].unique())
        contact_type = st.selectbox('Contact Communication Type', df['ContactCommunicationType'].unique())
        last_contact_month = st.selectbox('Last Contact Month', df['LastContactMonth'].unique())
        last_contact_day = st.selectbox('Last Contact Day', df['LastContactDayOfWeek'].unique())
        campaign_outcome = st.selectbox('Previous Campaign Outcome', df['PreviousCampaignOutcome'].unique())

    # Sağ kolon girişleri
    with col2:
        age = st.number_input('Age', df.Age.min(), df.Age.max())
        call_duration = st.number_input('Call Duration', df.CallDuration.min(), df.CallDuration.max())
        campaign_contacts = st.number_input('Campaign Contacts', df.CampaignContacts.min(), df.CampaignContacts.max())
        previous_contacts = st.number_input('Previous Campaign Contacts', df.PreviousCampaignContacts.min(), df.PreviousCampaignContacts.max())
        emp_var_rate = st.number_input('Employment Variation Rate', df['EmploymentVariationRate'].min(), df['EmploymentVariationRate'].max())
        consumer_price_idx = st.number_input('Consumer Price Index', df['ConsumerPriceIndex'].min(), df['ConsumerPriceIndex'].max())
        consumer_conf_idx = st.number_input('Consumer Confidence Index', df['ConsumerConfidenceIndex'].min(), df['ConsumerConfidenceIndex'].max())
        euribor_rate = st.number_input('Euribor 3M Rate', df['Euribor3MRate'].min(), df['Euribor3MRate'].max())
        num_employees = st.number_input('Number of Employees', df['NumberOfEmployees'].min(), df['NumberOfEmployees'].max())

    # Giriş verilerinden bir DataFrame oluşturma
    new_data = {
        'JobType': job_type,
        'MaritalStatus': marital_status,
        'EducationLevel': education_level,
        'HasHousingLoan': has_housing_loan,
        'HasPersonalLoan': has_personal_loan,
        'ContactCommunicationType': contact_type,
        'LastContactMonth': last_contact_month,
        'LastContactDayOfWeek': last_contact_day,
        'PreviousCampaignOutcome': campaign_outcome,
        'Age': age,
        'CallDuration': call_duration,
        'CampaignContacts': campaign_contacts,
        'PreviousCampaignContacts': previous_contacts,
        'EmploymentVariationRate': emp_var_rate,
        'ConsumerPriceIndex': consumer_price_idx,
        'ConsumerConfidenceIndex': consumer_conf_idx,
        'Euribor3MRate': euribor_rate,
        'NumberOfEmployees': num_employees
    }
    new_data = pd.DataFrame(new_data, index=[0])

    # Veriyi işleyici ile dönüştürme
    new_data_processed = preprocessor.transform(new_data)

    # Tahmin
    prediction = model.predict(new_data_processed)[0]
    prediction_text = 'YES' if prediction == 1 else 'NO'

    # Tahmin sonucunu gösterme
    if st.button('Predict'):
        st.markdown(f'# Deposit Prediction: {prediction_text}')
