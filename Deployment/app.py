import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import base64

# --- CONFIG
st.set_page_config(page_title="Delivery Delay Predictor", layout="centered")

# --- BACKGROUND & STYLING
def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    encoded_img = base64.b64encode(img_data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_img}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }}
        .block-container {{
            background-color: rgba(0, 0, 0, 0.75);
            padding: 2.5rem;
            border-radius: 1.5rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }}
        .stSelectbox, .stNumberInput, .stSlider {{
            margin-bottom: 1rem !important;
            box-shadow: 0 2px 6px rgba(255, 255, 255, 0.1);
        }}
        h2 {{
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 1.2rem;
        }}
        .stButton > button {{
            background-color: #00b894;
            color: white;
            border-radius: 6px;
        }}
        .top-right-logo {{
            position: absolute;
            top: 1.5rem;
            right: 2rem;
            z-index: 100;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- LOGO IN TOP RIGHT
def add_logo_top_right(logo_path, width=120):
    with open(logo_path, "rb") as image_file:
        encoded_logo = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <div class="top-right-logo">
            <img src="data:image/png;base64,{encoded_logo}" width="{width}">
        </div>
        """,
        unsafe_allow_html=True
    )

# --- APPLY BACKGROUND & LOGO
set_background("6.jpg")
add_logo_top_right("logo2w.png")

# --- LOAD MODEL
rf_model = joblib.load("random_forest_model.pkl")
preprocessor = joblib.load("prioprocessor.pkl")

# --- HEADER
st.title("Delay Detector")
st.markdown("AI-powered Logistics Intelligence.")

# --- FORM
with st.form("input_form"):
    st.subheader("📦 Enter Shipment Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        warehouse_block = st.selectbox("🏢 Warehouse Block", ["A", "B", "C", "D", "F"])
        customer_rating = st.slider("⭐ Customer Rating", 1, 5, 3)
        product_importance = st.selectbox("📊 Product Importance", ["low", "medium", "high"])
        weight_category = st.selectbox("⚖️ Weight Category", ["low", "medium", "high"])

    with col2:
        mode_of_shipment = st.selectbox("✈️ Mode of Shipment", ["Flight", "Ship", "Road"])
        cost_of_the_product = st.number_input("💲 Cost of the Product", 50, 500, 250)
        gender = st.selectbox("🧍 Customer Gender", ["M", "F"])
        discount_offered = st.number_input("🏷️ Discount Offered", 0, 65, 10)

    with col3:
        customer_care_calls = st.slider("📞 Customer Care Calls", 0, 10, 3)
        prior_purchases = st.slider("🔁 Prior Purchases", 0, 10, 2)

    input_data = pd.DataFrame([{
        "warehouse_block": warehouse_block,
        "mode_of_shipment": mode_of_shipment,
        "customer_care_calls": customer_care_calls,
        "customer_rating": customer_rating,
        "cost_of_the_product": cost_of_the_product,
        "prior_purchases": prior_purchases,
        "product_importance": product_importance,
        "gender": gender,
        "discount_offered": discount_offered,
        "weight_category": weight_category
    }])

    with st.expander("📄 Review Shipment Data", expanded=True):
        st.dataframe(input_data)

    submitted = st.form_submit_button("Predict")

# --- PREDICTION
if submitted:
    input_preprocessed = preprocessor.transform(input_data)
    probability = rf_model.predict_proba(input_preprocessed)[:, 1]
    prediction = (probability >= 0.40).astype(int)[0]

    if prediction == 1:
        label = "⚠️ Late Delivery (1)"
        color = "red"
        bar_color = "red"
    else:
        label = "✅ On Time Delivery (0)"
        color = "lightgreen"
        bar_color = "darkgreen"

    st.subheader("📊 Prediction Result")
    st.markdown(f"**Predicted Probability of Being Late:** `{probability[0]:.2f}`")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability[0] * 100,
        number={'suffix': '%', 'font': {'size': 48}},
        title={'text': "Risk of Delay", 'font': {'size': 22}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
            'bar': {'color': bar_color},
            'steps': [
                {'range': [0, 40], 'color': "#32CD32"},
                {'range': [40, 100], 'color': "lightcoral"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 40
            }
        }
    ))

    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"<h3 style='text-align: center; color: {color};'>{label}</h3>", unsafe_allow_html=True)
