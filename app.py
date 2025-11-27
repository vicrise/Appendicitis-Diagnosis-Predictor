# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ------------------- FIX: Correct path -------------------
DATA_PATH = "cleaned_data.csv"

# ------------------- Page config -------------------
st.set_page_config(
    page_title="Appendicitis Diagnosis Predictor",
    page_icon="hospital",
    layout="centered"
)

st.title("Appendicitis Diagnosis Predictor")
st.markdown("This app predicts the type of appendicitis based on clinical signs and symptoms using a trained XGBoost model")

# ------------------- Train & cache model once -------------------
@st.cache_resource
def train_and_get_artifacts():
    # Load data
    df = pd.read_csv(DATA_PATH)

    # Features & target
    X = df.drop(columns=["Management", "Diagnosis", "Severity"])
    X = pd.get_dummies(X, drop_first=False)
    y_raw = df["Diagnosis"]

    # Label encode target
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Remove classes with only 1 sample (prevents SMOTE crash)
    value_counts = pd.Series(y).value_counts()
    rare_classes = value_counts[value_counts < 2].index
    if len(rare_classes) > 0:
        mask = ~pd.Series(y).isin(rare_classes)
        X = X[mask].reset_index(drop=True)
        y = y[mask]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE with safe k_neighbors
    min_count = pd.Series(y_train).value_counts().min()
    k_neighbors = min(5, min_count - 1) if min_count > 1 else 1

    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)

    # Train model
    model = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_estimators=300,
        learning_rate=0.1
    )
    model.fit(X_train_scaled, y_train_res)

    # Save everything for future fast loading (optional but recommended)
    joblib.dump(model, "xgb_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(le, "label_encoder.pkl")
    joblib.dump(X.columns.tolist(), "feature_names.pkl")

    return model, scaler, le, X.columns

# Load (or train once) all artifacts
model, scaler, le, feature_names = train_and_get_artifacts()



# ========================= USER INPUT FORM =========================
def user_input_features():
    st.sidebar.header("Patient Clinical Features")

    # --- Demographic ---
    with st.sidebar.subheader("Demographics"):
        Age = st.sidebar.slider("Age", 1, 100, 12)
        BMI = st.sidebar.number_input("BMI", min_value=7.00, max_value=50.0, value=18.0, step=0.1)
        Sex = st.sidebar.selectbox("Sex", options=["male", "female"])
        Height = st.sidebar.number_input("Height", min_value=50, max_value=200, value=50)
        Weight = st.sidebar.number_input("Weight", min_value=2, max_value=200, value=18)


    # --- Symptoms ---
    with st.sidebar.subheader("Symptoms"):
        Migratory_Pain = st.sidebar.selectbox("Migratory_Pain", ["yes", "no","unknown"])
        Lower_Right_And_Pain = st.sidebar.selectbox("Lower Right Abdominal Pain", ["yes", "no","unknown"])
        Loss_of_Appetite = st.sidebar.selectbox("Loss of Appetite", ["yes", "no","unknown"])
        Nausea = st.sidebar.selectbox("Nausea", ["yes", "no","unknown"])
        Coughing_Pain = st.sidebar.selectbox("Coughing", ["yes", "no","unknown"])
        Contralateral_Rebound_Tenderness = st.sidebar.selectbox("C ontralateralRebound Tenderness", ["yes", "no","unknown"])
        Psoas_Sign = st.sidebar.selectbox("Psoas Sign", ["yes", "no","unknown"])
        Ipsilateral_Rebound_Tenderness = st.sidebar.selectbox("Ipsilateral Rebound Tenderness", ["yes", "no","unknown"])
        Diagnosis_Presumptive = st.sidebar.selectbox("Diagnosis_Presumptive", ["Appendicitis", "Appendicitis With Mesenteric Lymph Node Inflammation","No Appendicitis","Gastroenteritis","Chronic Appendicitis",
                                                                           "Ovarian Torsion","Prolonged Gastroenteritis","Diabetic Ketoacidosis With Myocarditis","Chronic Abdominal Pain","Sepsis With Accompanying Appendicitis",
                                                                           "Adnexal Torsion","Abdominal Adhesions With Partial Bowel Obstruction","Adhesions Of The Ascending Colon","Adhesive Ileus","Perforated Appendicitis"])

    # --- Clinical Signs ---
    with st.sidebar.subheader("Clinical & Lab Findings"):
        Body_Temperature = st.sidebar.slider("Body Temperature", 20.0, 41.0, 37.0, step=0.1)
        RBC_Count = st.sidebar.number_input("RBC Count (×10⁹/L)", 2.0, 40.0, 12.0, step=0.1)
        Neutrophil_Percentage = st.sidebar.slider("Neutrophil %", 20.0, 100.0, 50.0, step=0.5)
        Neutrophilia = st.sidebar.selectbox("Neutrophilia", ["yes", "no","unknown"])
        CRP = st.sidebar.number_input("CRP (mg/L)", 0, 500, 10, step=10)
        Alvarado_Score = st.sidebar.number_input("Alvarado_Score", min_value=0, max_value=10, value=1)
        Paedriatic_Appendicitis_Score = st.sidebar.number_input("Paedriatic_Appendicitis_Score", min_value=0, max_value=10, value=1)
        Peritonitis = st.sidebar.selectbox("Peritonitis", ["generalized","local", "no","unknown"])


    # --- Ultrasound Findings ---
    with st.sidebar.subheader("Ultrasound Findings"):
        US_Performed = st.sidebar.selectbox("Was Ultrasound Performed?", ["yes", "no","unknown"])
        Appendix_on_US = st.sidebar.selectbox("Appendix Visualized on US", ["yes", "no","unknown"])
        Appendix_Diameter = st.sidebar.number_input("Appendix Diameter (mm)", 0.0, 20.0, 7.5, step=0.1)
        Free_Fluids = st.sidebar.selectbox("Free Fluid in Abdomen", ["yes", "no","unknown"])
        Appendicolith = st.sidebar.selectbox("Appendicolith (Fecalith)", ["yes", "no","unknown"])
        Target_Sign = st.sidebar.selectbox("Target Sign on US", ["yes", "no"])
        US_Number = st.sidebar.number_input("US_Number", 1, 1000, 20)
        Appendix_Wall_Layers = st.sidebar.selectbox("Appendix_Wall_Layers", ["intact", "raised","unknown","partially raised","upset"])
        Perfusion = st.sidebar.selectbox("Perfusion", ["hyperperfused", "hypoperfused", "unknown","no","present"])
        Perforation = st.sidebar.selectbox("Perforation Signs", ["yes", "no", "not excluded", "unknown","suspected"])
        Surrounding_Tissue_Reaction = st.sidebar.selectbox("Surrounding Tissue Reaction", ["yes", "no", "unknown"])
        Appendicular_Abscess = st.sidebar.selectbox("Appendicular Abscess", ["yes", "no", "unknown","suscepted"])
        Abscess_Location = st.sidebar.selectbox("Abscess Location",["Unknown","Pelvic Cavity","Behind The Bladder","Right Lower Abdomen",
                                                                 "Around The Cecum","Right Psoas Muscle Region","Right Mid-Abdomen"])
        Pathological_Lymph_Nodes = st.sidebar.selectbox("Pathological Lymph Nodes", ["yes", "no", "unknown"])
        Lymph_Nodes_Location = st.sidebar.selectbox("Lymph Nodes Location", ["Right lower abdomen","unknown","Ileocecal","Lower abdomen","Periumbilical","Mesenteric and right lower abdomen",
                                                                          "Mesenteric","Right lower abdomen and periumbilical","re UB","Right lower abdomen and ileocecal","Right lower and middle abdomen",
                                                                          "Middle abdomen","Right middle abdomen","Inguinal","Periappendiceal","Lymphadenopathy","Mesenteric and left inguinal","Multiple locations","Around the appendix","Ovarian cysts"])
        Bowel_Wall_Thickening = st.sidebar.selectbox("Bowel Wall Thickening", ["yes", "no", "unknown"])
        Conglomerate_of_Bowel_Loops = st.sidebar.selectbox("Conglomerate of Bowel Loops", ["yes", "no", "unknown"])
        Ileus = st.sidebar.selectbox("Ileus", ["yes", "no", "unknown"])
        Coprostasis = st.sidebar.selectbox("Coproostasis", ["yes", "no", "unknown"])
        Meteorism = st.sidebar.selectbox("Meteorism", ["yes", "no", "unknown"])
        Enteritis = st.sidebar.selectbox("Enteritis", ["yes", "no", "unknown"])
        Gynecological_Findings = st.sidebar.selectbox("Gynecological Findings", ["unknown","Ovarian cyst","Uterine cyst","Bilateral ovarian cysts with abnormal right ovary perfusion",
                                                                              "Normal ovaries","Right ovarian cyst","No gynecological cause","Suspected ovarian torsion","Yes",
                                                                            "Ovarian cysts","Normal finding"])

    # --- Other ---
    with st.sidebar.subheader("Other"):
        Lenght_of_Stay = st.sidebar.number_input("Lenght_of_Stay", min_value=1, max_value=30, value=18, step=10)
        Dysuria = st.sidebar.selectbox("Dysuria", ["yes", "no","unknown"])
        Stool = st.sidebar.selectbox("Stool Pattern", ["normal", "diarrhea", "constipation","constipationand diarrhea","unknown"])
        Hemoglobin = st.sidebar.number_input("Hemoglobin", 1.0, 40.0, 7.5, step=0.1)
        RDW  = st.sidebar.number_input("RDW", 10.0, 100.0, 20.0, step=0.1)
        Thrombocyte_Count =  st.sidebar.number_input("Thrombocyte_Count", 90, 800, 100)
        Ketones_in_Urine = st.sidebar.selectbox("Ketones_in_Urine", ["+", "++", "++","no","unknown"])
        RBC_in_Urine = st.sidebar.selectbox("RBC_in_Urine", ["+", "++", "++","no","unknown"])
        WBC_in_Urine = st.sidebar.selectbox("WBC_in_Urine", ["+", "++", "++","no","unknown"])
   

    # Create dictionary
    data = {
        "Age": Age,
        "BMI": BMI,
        "Sex": Sex,
        "Height": Height,
        "Weight": Weight,
        "Migratory_Pain": Migratory_Pain,
        "Lower_Right_And_Pain": Lower_Right_And_Pain,
        "Contralateral_Rebound_Tenderness": Contralateral_Rebound_Tenderness,
        "Coughing_Pain": Coughing_Pain,
        "Nausea": Nausea,
        "Loss_of_Appetite": Loss_of_Appetite,
        "Body_Temperature": Body_Temperature,
        "Neutrophil_Percentage": Neutrophil_Percentage,
        "Neutrophilia": Neutrophilia,
        "RBC_Count": RBC_Count,
        "Hemoglobin": Hemoglobin,
        "RDW": RDW,
        "Thrombocyte_Count": Thrombocyte_Count,
        "Ketones_in_Urine": Ketones_in_Urine,
        "RBC_in_Urine": RBC_in_Urine,
        "WBC_in_Urine": WBC_in_Urine,
        "CRP": CRP,
        "Dysuria": Dysuria,
        "Stool": Stool,
        "Peritonitis": Peritonitis,
        "Psoas_Sign": Psoas_Sign,
        "Ipsilateral_Rebound_Tenderness": Ipsilateral_Rebound_Tenderness,
        "US_Performed": US_Performed,
        "US_Number": US_Number,
        "Appendix_on_US": Appendix_on_US,
        "Appendix_Diameter": Appendix_Diameter,
        "Free_Fluids": Free_Fluids,
        "Appendix_Wall_Layers": Appendix_Wall_Layers,
        "Target_Sign": Target_Sign,
        "Appendicolith": Appendicolith,
        "Perfusion": Perfusion,
        "Perforation": Perforation,
        "Surrounding_Tissue_Reaction": Surrounding_Tissue_Reaction,
        "Appendicular_Abscess": Appendicular_Abscess,
        "Abscess_Location": Abscess_Location,
        "Pathological_Lymph_Nodes": Pathological_Lymph_Nodes,
        "Lymph_Nodes_Location": Lymph_Nodes_Location,
        "Bowel_Wall_Thickening": Bowel_Wall_Thickening,
        "Conglomerate_of_Bowel_Loops": Conglomerate_of_Bowel_Loops,
        "Ileus": Ileus,
        "Coprostasis": Coprostasis,
        "Meteorism": Meteorism,
        "Enteritis": Enteritis,
        "Gynecological_Findings": Gynecological_Findings}
    return pd.DataFrame([data])



st.subheader("Enter Patient Details")
input_df = user_input_features()
def prepare_input(df):
    df_encoded = pd.get_dummies(df, drop_first=False)
    df_aligned = df_encoded.reindex(columns=feature_names, fill_value=0)
    X_scaled = scaler.transform(df_aligned)
    return X_scaled

if st.sidebar.button("Predict Diagnosis"):
    X_input = prepare_input(input_df)
    with st.spinner("Analyzing..."):
        prediction = model.predict(X_input)
        probabilities = model.predict_proba(X_input)[0]
    predicted_diagnosis = le.inverse_transform(prediction)[0]
    confidence = max(probabilities) * 100
    st.success(f"**Predicted Diagnosis: {predicted_diagnosis}**")
    st.info(f"Confidence: {confidence:.1f}%")
    
    prob_df = pd.DataFrame({
        "Diagnosis": le.classes_,
        "Probability (%)": np.round(probabilities * 100, 2)
    }).sort_values("Probability (%)", ascending=False)
    st.write("### All Class Probabilities")
    st.dataframe(prob_df, use_container_width=True)
    if predicted_diagnosis == "appendicitis":
        st.error("URGENT: This may require immediate surgery!")
    else:
        st.info("No urgent action required based on prediction.")
else:
    st.info("Adjust patient details in the sidebar → Click **Predict Diagnosis**")

# Footer
st.markdown("---")

st.caption("XGBoost + SMOTE • Trained on real clinical data • Built with Streamlit")
