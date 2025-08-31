import pickle
import os
import pickle
import pandas as pd




import streamlit as st
from streamlit_option_menu import option_menu
import fitz  # PyMuPDF

# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# Load models
working_dir = os.path.dirname(os.path.abspath(__file__))

diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))
# Load Kidney model and scaler
kidney_scaler = pickle.load(open(f'{working_dir}/saved_models/scaler.pkl', 'rb'))
kidney_model = pickle.load(open(f'{working_dir}/saved_models/model_gbc.pkl', 'rb'))

# Sidebar menu
# Sidebar menu update
with st.sidebar:
    selected = option_menu('DORAEMON MINI-DOCTOR',
                           ['Home','Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Insurance Prediction', 'Kidney Disease Prediction'],
                           icons=['house-door-fill','activity', 'heart-pulse', 'person-wheelchair', 'file-earmark-text', 'capsule'],
                           menu_icon='hospital-fill', default_index=0)

# ---------------- PDF Upload for Health Predictions ----------------
def extract_pdf_data(uploaded_pdf):
    doc = fitz.open(uploaded_pdf)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


#-----------------HOME PAGE-------------#
if selected == 'Home':
    st.title("🏥 Welcome to the DORAEMON MINI-DOCTOR")
    st.markdown("""
    Welcome to **Health Assistant**, your intelligent tool powered by Machine Learning to help predict and assess various health conditions.

    ### 🔍 What can you do here?

    ✴️ Predict:
    - 🩸 **Diabetes**
    - ❤️ **Heart Disease**
    - 🧠 **Parkinson's Disease**
    - 🧪 **Kidney Disease**

    ✴️ Estimate your **Medical Insurance Cost**  
    ✴️ Upload and extract data from **PDF medical reports**  
    ✴️ Get instant results through a user-friendly form  

    ---

    ### 🚀 Powered By:
    - 🤖 **Machine Learning Models**
    - 🐍 **Python**
    - 🖥️ **Streamlit**
    - 📄 **PyMuPDF** for PDF extraction

    ---

    👨‍⚕️ *Note: This system is built to assist, not replace, a professional medical diagnosis.*

    ---
    
    ### 🔮 Future Features:
    
    🔹 **Symptom-Based Disease Prediction
                
    🔹 **Lab Report Parser**: Upload and extract values from medical PDFs and images with OCR. 
                 
    🔹 **Health Risk Score Dashboard**
                
    🔹 **Health Tracker Integration**: Track health over time through trends like blood pressure or hemoglobin levels.  
                
    🔹 **Smart Lifestyle Recommendations**:
                
    🔹 **Add More Diseases**: Add liver disease, breast cancer, anemia, and more predictions.  
  
    🔹 **Multilingual Support**: Translate the app into multiple languages (Hindi, Kannada, Turkish, etc.).  
                
    🔹 **Chatbot Assistant**: Ask health-related questions and receive instant answers.  
    """)



# ---------------- Diabetes Page ----------------
if selected == 'Diabetes Prediction':
    st.title('🩸 Diabetes Prediction ')

    # PDF Upload and Data Extraction
    uploaded_pdf = st.file_uploader("Upload Medical Report (PDF)", type=["pdf"])
    if uploaded_pdf is not None:
        extracted_text = extract_pdf_data(uploaded_pdf)
        st.text_area("Extracted Text", extracted_text, height=300)

    with st.expander("📝 Fill the input data manually"):
        col1, col2, col3 = st.columns(3)
        with col1:
            Pregnancies = st.number_input('Number of Pregnancies', min_value=0.0)
            SkinThickness = st.number_input('Skin Thickness value', min_value=0.0)
            DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0)
        with col2:
            Glucose = st.number_input('Glucose Level', min_value=0.0)
            Insulin = st.number_input('Insulin Level', min_value=0.0)
            Age = st.number_input('Age of the Person', min_value=0)
        with col3:
            BloodPressure = st.number_input('Blood Pressure value', min_value=0.0)
            BMI = st.number_input('BMI value', min_value=0.0)

    if st.button('🔍 Get Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        diab_prediction = diabetes_model.predict([user_input])
        result = '✅ The person is not diabetic' if diab_prediction[0] == 0 else '⚠️ The person is diabetic'
        st.success(result)
        # ---------------- Kidney Disease Page ----------------
if selected == "Kidney Disease Prediction":
    st.title("🧬 Kidney Disease Prediction using ML")
    st.markdown("### Check if you might be at risk of Chronic Kidney Disease (CKD)")

    # Input form (reuse the code from your Kidney app)
    with st.form("kidney_form"):
        age = st.slider("Age", 1, 100, 30)
        bp = st.slider("Blood Pressure (in mm/Hg)", 50, 200, 80)
        sg = st.select_slider("Specific Gravity (sg)", options=[1.005, 1.010, 1.015, 1.020, 1.025, 1.030], value=1.020)
        al = st.slider("Albumin Level", 0.0, 5.0, 1.0, 0.1)
        hemo = st.slider("Hemoglobin (g/dl)", 3.0, 17.5, 12.0)
        sc = st.slider("Serum Creatinine (mg/dl)", 0.5, 5.0, 1.2)

        htn = st.selectbox("Do you have Hypertension?", ("yes", "no"))
        dm = st.selectbox("Do you have Diabetes?", ("yes", "no"))
        cad = st.selectbox("Do you have Coronary Artery Disease?", ("yes", "no"))
        appet = st.selectbox("How is your appetite?", ("good", "poor"))
        pc = st.selectbox("Protein content in urine", ("normal", "abnormal"))

        submitted = st.form_submit_button("🔍 Predict CKD")

    # On submit, process the data and make the prediction
    if submitted:
        df_dict = {
            'age': [age],
            'bp': [bp],
            'sg': [sg],
            'al': [al],
            'hemo': [hemo],
            'sc': [sc],
            'htn': [1 if htn == "yes" else 0],
            'dm': [1 if dm == "yes" else 0],
            'cad': [1 if cad == "yes" else 0],
            'appet': [1 if appet == "good" else 0],
            'pc': [1 if pc == "normal" else 0]
        }

        df = pd.DataFrame(df_dict)

        # Scale the numeric features
        numeric_cols = ['age', 'bp', 'sg', 'al', 'hemo', 'sc']
        df[numeric_cols] = kidney_scaler.transform(df[numeric_cols])

        # Prediction
        result = kidney_model.predict(df)[0]

        # Output result
        if result == 1:
            st.error("⚠️ The patient is at risk of **Chronic Kidney Disease (CKD)**.")
        else:
            st.success("✅ The patient is **not likely** to have CKD.")

        st.markdown("---")

# ---------------- Heart Disease Page ----------------
if selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction using ML')

    # PDF Upload and Data Extraction
    uploaded_pdf = st.file_uploader("Upload Medical Report (PDF)", type=["pdf"])
    if uploaded_pdf is not None:
        extracted_text = extract_pdf_data(uploaded_pdf)
        st.text_area("Extracted Text", extracted_text, height=300)

    with st.expander("📝 Fill the input data manually"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input('Age', min_value=0.0)
            trestbps = st.number_input('Resting Blood Pressure', min_value=0.0)
            restecg = st.number_input('Resting Electrocardiographic results')
            oldpeak = st.number_input('ST depression induced by exercise')
            thal = st.number_input('thal (0=normal; 1=fixed defect; 2=reversable defect)')
        with col2:
            sex = st.number_input('Sex (1=male, 0=female)', min_value=0.0, max_value=1.0)
            chol = st.number_input('Serum Cholesterol in mg/dl', min_value=0.0)
            thalach = st.number_input('Maximum Heart Rate achieved', min_value=0.0)
            slope = st.number_input('Slope of peak exercise ST segment')
        with col3:
            cp = st.number_input('Chest Pain type (0–3)', min_value=0.0)
            fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)', min_value=0.0, max_value=1.0)
            exang = st.number_input('Exercise Induced Angina (1=Yes, 0=No)', min_value=0.0, max_value=1.0)
            ca = st.number_input('Number of major vessels (0-3) colored by flourosopy')

    if st.button('🔍 Get Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]
        heart_prediction = heart_disease_model.predict([user_input])
        result = '✅ The person does not have any heart disease' if heart_prediction[0] == 0 else '⚠️ The person has heart disease'
        st.success(result)
        

# ---------------- Parkinson's Page ----------------
if selected == "Parkinsons Prediction":
    st.title("🧠 Parkinson's Disease Prediction using ML")

    # PDF Upload and Data Extraction
    uploaded_pdf = st.file_uploader("Upload Medical Report (PDF)", type=["pdf"])
    if uploaded_pdf is not None:
        extracted_text = extract_pdf_data(uploaded_pdf)
        st.text_area("Extracted Text", extracted_text, height=300)

    with st.expander("📝 Fill the input data manually"):
        cols = st.columns(5)
        fields = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
            'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
            'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR',
            'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
        ]

        user_values = []
        for i, field in enumerate(fields):
            with cols[i % 5]:
                val = st.number_input(field)
                user_values.append(val)

    if st.button("🔍 Get Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([user_values])
        result = "✅ The person does not have Parkinson's disease" if parkinsons_prediction[0] == 0 else "⚠️ The person has Parkinson's disease"
        st.success(result)

# ---------------- Insurance Prediction Page ----------------
if selected == "Insurance Prediction":
    st.title("💰 Medical Insurance Cost Prediction")
    with st.expander("📝 Fill the input data"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=0)
            sex = st.radio("Gender", ("Male", "Female"))
        with col2:
            bmi = st.number_input("BMI", min_value=0.0)
            children = st.number_input("Number of Children", min_value=0)
        with col3:
            smoker = st.radio("Smoker", ("Yes", "No"))
            region = st.selectbox("Region", ["Southeast", "Southwest", "Northeast", "Northwest"])

    # Converting inputs to the format required for prediction
    sex = 1 if sex == "Male" else 0
    smoker = 1 if smoker == "Yes" else 0
    region_map = {"Southeast": 0, "Southwest": 1, "Northeast": 2, "Northwest": 3}
    region = region_map[region]

    if st.button("🔍 Predict Insurance Cost"):
        user_input = [age, sex, bmi, children, smoker, region]
        # Assuming you have a trained model for insurance prediction
        # Replace insurance_model with your actual model
        insurance_model = pickle.load(open(f'{working_dir}/saved_models/insurance_model.pkl', 'rb'))
        insurance_prediction = insurance_model.predict([user_input])
        st.success(f"💰 The predicted insurance cost is: ${insurance_prediction[0]:.2f}")
        # st.success(f"💰 The predicted insurance cost is: ₹{insurance_prediction[0] * 80:.2f}")