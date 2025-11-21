import streamlit as st
import pandas as pd
import pickle
import streamlit.components.v1 as components

# ====================================
# Load artifact
# ====================================
with open("heart_artifact.pkl", "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
scaler = artifact["scaler"]
columns = artifact["columns"]
num_cols = artifact["num_cols"]  # <-- ADDED

# ====================================
# PAGE CONFIG
# ====================================
st.set_page_config(
    page_title="Heart Disease Predictor",
    layout="wide",
    page_icon="❤️",
)

# ====================================
# CUSTOM PAGE HEADER
# ====================================
st.markdown("""
    <div style="text-align:center; margin-bottom:20px;">
        <h1 style="color:#ff4b4b;">❤️ Heart Disease Prediction</h1>
        <p style="font-size:18px; color:white;">Enter patient details below to assess risk</p>
    </div>
""", unsafe_allow_html=True)

# ====================================
# MAPPINGS
# ====================================
cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}
restecg_map = {
    "Normal": 0,
    "ST-T Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}
slope_map = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}
thal_map = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

# ====================================
# PATIENT INPUT SECTION
# ====================================
st.subheader("🧍 Patient Information")

st.markdown("### 📌 General Information")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 20, 100, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])

with col2:
    trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 240)

with col3:
    thalach = st.number_input("Max Heart Rate", 60, 210, 150)
    oldpeak = st.number_input("ST Depression", 0.0, 6.5, 1.0, step=0.1)

st.markdown("---")

st.markdown("### 🩺 Clinical Symptoms")
col4, col5, col6 = st.columns(3)

with col4:
    cp_label = st.selectbox("Chest Pain Type", list(cp_map.keys()))
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])

with col5:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"])
    slope_label = st.selectbox("Slope of ST Segment", list(slope_map.keys()))

with col6:
    st.markdown("<div class='center-widget'>", unsafe_allow_html=True)
    restecg_label = st.selectbox("Resting ECG", list(restecg_map.keys()))
    ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
    thal_label = st.selectbox("Thalassemia Test", list(thal_map.keys()))

st.markdown("---")

# ====================================
# IDEAL HEALTH RANGES CARD (HTML COMPONENT)
# ====================================


html_code = """
<div style="
    max-width: 650px;
    margin: 20px auto;
    padding: 25px;
    background-color: #ffffff;
    border-radius: 15px;
    border: 2px solid #e00202;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.18);
">
    <h3 style='color:#e00202; text-align:center; margin-top:0; font-size:26px;'>
        ❤️ Ideal Heart Health Values
    </h3>

<ul style="
    font-size:16px; 
    color:#444; 
    line-height:1.6; 
    margin-left: 10px;
    padding-left: 15px;
    font-family: 'Source Sans Pro', sans-serif;
">
    <li><b>Resting BP:</b> 120/80 mm Hg</li>
    <li><b>Cholesterol:</b> Below 200 mg/dL</li>
    <li><b>Fasting Blood Sugar:</b> Less than 120 mg/dL</li>
    <li><b>ECG:</b> Normal (0)</li>
    <li><b>Max Heart Rate:</b> 140–190 bpm</li>
    <li><b>ST Depression:</b> 0 – 1.0</li>
    <li><b>Slope:</b> Upsloping (0)</li>
    <li><b>Major Vessels:</b> 0</li>
    <li><b>Thalassemia:</b> Normal (1)</li>
</ul>

</div>
"""
components.html(html_code, height=500)

# ====================================
# ENCODING USER INPUT
# ====================================
sex = 1 if sex == "Male" else 0
exang = 1 if exang == "Yes" else 0
fbs = 1 if fbs == "Yes" else 0

cp = cp_map[cp_label]
restecg = restecg_map[restecg_label]
slope = slope_map[slope_label]
thal = thal_map[thal_label]

# ====================================
# PREPROCESS FUNCTION
# ====================================
def preprocess():
    data = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg,
        "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "slope": slope, "ca": ca, "thal": thal
    }

    df = pd.DataFrame([data])
    df = pd.get_dummies(df, columns=["cp","restecg","slope","thal"], drop_first=True)

    # Add missing one-hot columns
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    # ORDER columns correctly
    df = df[columns]

    # SCALE numeric columns correctly
    df[num_cols] = scaler.transform(df[num_cols])   # <-- FIXED

    return df

# ====================================
# PREDICTION UI
# ====================================
st.markdown("### 🔮 Prediction")

if st.button("Predict Risk", use_container_width=True):

    processed = preprocess()
    pred = model.predict(processed)[0]
    prob = model.predict_proba(processed)[0][1] * 100

    st.markdown("---")

    card_style = """
        <div style="
            padding:28px;
            border-radius:15px;
            margin-top:10px;
            background-color:#ffffff;
            border: 2px solid {border_color};
        ">
            <h2 style="color:{title_color}; margin-bottom:10px;">{title}</h2>
            <p style="font-size:22px; color:#333;"><b>Probability: {prob:.2f}%</b></p>
            <p style="font-size:16px; color:#555;">{message}</p>
        </div>
    """

    if pred == 1:
        card = card_style.format(
            border_color="#ff4b4b",
            title_color="#cc0000",
            title="❤️ High Risk of Heart Disease",
            prob=prob,
            message="Immediate medical consultation is recommended."
        )
    else:
        card = card_style.format(
            border_color="#00b300",
            title_color="#008000",
            title="💚Low Risk",
            prob=prob,
            message="Keep maintaining a healthy lifestyle."
        )

    st.markdown(card, unsafe_allow_html=True)
