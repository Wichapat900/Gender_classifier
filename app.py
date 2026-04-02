import streamlit as st
import pandas as pd
import pickle

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(page_title="Gender Classifier", page_icon="🎵", layout="centered")

st.title("🎵 Gender Classifier")
st.markdown("Fill in the details below and the model will predict the gender!")
st.divider()

# ── Load Model ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('gender_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model file not found! Make sure 'gender_model.pkl' is in the same folder as this app.")
    st.stop()

st.divider()

# ── Input Form ────────────────────────────────────────────────────────
st.subheader("👤 Enter Person Details")

col1, col2 = st.columns(2)

with col1:
    skirt = st.radio(
        "👗 Wearing a skirt?",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    hair = st.radio(
        "💇 Hair longer than shoulder?",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col2:
    frequency = st.slider(
        "🎵 Speaking Frequency (Hz)",
        min_value=80,
        max_value=400,
        value=200,
        step=1,
        
    )



st.divider()

# ── Predict Button ────────────────────────────────────────────────────
if st.button("🔍 Predict Gender", use_container_width=True, type="primary"):
    input_data = pd.DataFrame({
        'skirt': [skirt],
        'hair': [hair],
        'frequency': [frequency]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    confidence = max(probability) * 100

    st.divider()
    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.markdown(f"## 👧 Girl")
        st.success(f"Confidence: **{confidence:.1f}%**")
    else:
        st.markdown(f"## 👦 Boy")
        st.info(f"Confidence: **{confidence:.1f}%**")

    # Show input summary
    with st.expander("📋 Input Summary"):
        st.write(f"- **Skirt:** {'Yes' if skirt else 'No'}")
        st.write(f"- **Long Hair:** {'Yes' if hair else 'No'}")
        st.write(f"- **Frequency:** {frequency} Hz")

    # Probability bar
    st.markdown("**Prediction Probabilities:**")
    prob_df = pd.DataFrame({
        'Gender': ['Boy 👦', 'Girl 👧'],
        'Probability': [probability[0], probability[1]]
    })
    st.bar_chart(prob_df.set_index('Gender'))