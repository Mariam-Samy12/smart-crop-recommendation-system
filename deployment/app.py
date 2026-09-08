import streamlit as st
from tensorflow import keras
import joblib
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(os.path.join(BASE_DIR, "best_crop_mlp.keras"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
    return model, scaler, label_encoder


model, scaler, label_encoder = load_artifacts()


# Page settings
st.set_page_config(
    page_title="Smart Crop Advisor",
    page_icon="🌱",
    layout="centered"
)


# Simple styling
st.markdown("""
<style>

.stApp {
    background-color: #f4f8f1;
}

.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1 {
    text-align: center;
    color: #28552a;
    font-size: 40px;
}

.subtitle {
    text-align: center;
    color: #61735f;
    font-size: 17px;
    margin-bottom: 30px;
}

.section-title {
    color: #28552a;
    font-size: 24px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

.info-box {
    background-color: #eaf3e5;
    padding: 16px;
    border-radius: 14px;
    margin-top: 20px;
}

.result-box {
    background-color: white;
    padding: 16px;
    border-radius: 14px;
    margin-bottom: 10px;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.06);
}

.crops-box {
    background-color: white;
    padding: 16px;
    border-radius: 14px;
    line-height: 2;
}

</style>
""", unsafe_allow_html=True)


# Title
st.title("🌱 Smart Crop Advisor")

st.markdown(
    '<div class="subtitle">'
    'Enter your soil and weather conditions to find suitable crops.'
    '</div>',
    unsafe_allow_html=True
)


# Input section
st.markdown(
    '<div class="section-title">🌍 Soil & Weather Conditions</div>',
    unsafe_allow_html=True
)


# First row
col1, col2, col3 = st.columns(3)

with col1:
    nitrogen = st.number_input(
        "🌿 Nitrogen (N)",
        min_value=0.0,
        max_value=200.0,
        value=None,
        step=1.0,
        placeholder="e.g. 90"
    )

with col2:
    phosphorus = st.number_input(
        "🌱 Phosphorus (P)",
        min_value=0.0,
        max_value=200.0,
        value=None,
        step=1.0,
        placeholder="e.g. 42"
    )

with col3:
    potassium = st.number_input(
        "🪴 Potassium (K)",
        min_value=0.0,
        max_value=250.0,
        value=None,
        step=1.0,
        placeholder="e.g. 43"
    )


# Second row
col4, col5, col6, col7 = st.columns(4)

with col4:
    temperature = st.number_input(
        "🌡️ Temperature (°C)",
        min_value=-10.0,
        max_value=60.0,
        value=None,
        step=0.1,
        placeholder="e.g. 25.5"
    )

with col5:
    humidity = st.number_input(
        "💧 Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=None,
        step=1.0,
        placeholder="e.g. 80"
    )

with col6:
    ph = st.number_input(
        "🧪 Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=None,
        step=0.1,
        placeholder="e.g. 6.5"
    )

with col7:
    rainfall = st.number_input(
        "🌧️ Rainfall (mm)",
        min_value=0.0,
        max_value=1000.0,
        value=None,
        step=1.0,
        placeholder="e.g. 200"
    )


st.write("")


# Prediction button
predict_button = st.button(
    "🌾 Get Crop Recommendations",
    use_container_width=True
)


# Prediction process
if predict_button:

    # Check for empty inputs
    inputs = {
        "Nitrogen": nitrogen,
        "Phosphorus": phosphorus,
        "Potassium": potassium,
        "Temperature": temperature,
        "Humidity": humidity,
        "Soil pH": ph,
        "Rainfall": rainfall
    }

    missing_inputs = [
        name
        for name, value in inputs.items()
        if value is None
    ]

    if missing_inputs:

        st.warning(
            "Please enter all values before getting recommendations."
        )

    else:

        # Prepare data for FastAPI
        data = {
            "N": nitrogen,
            "P": phosphorus,
            "K": potassium,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }

        try:

            input_data = [[
                data["N"],
                data["P"],
                data["K"],
                data["temperature"],
                data["humidity"],
                data["ph"],
                data["rainfall"]
            ]]

            input_scaled = scaler.transform(input_data)

            probabilities = model.predict(input_scaled, verbose=0)[0]

            sorted_indices = probabilities.argsort()[::-1]

            threshold = 0.01

            predictions = []

            for index in sorted_indices:
                confidence = probabilities[index]

                if confidence >= threshold:
                    crop = label_encoder.inverse_transform([index])[0]

                    predictions.append({
                        "crop": crop,
                        "confidence": round(float(confidence * 100), 2)
                    })

            st.markdown(
                '<div class="section-title">'
                '🌾 Recommended Crops'
                '</div>',
                unsafe_allow_html=True
            )

            if len(predictions) == 0:

                st.warning(
                    "No crop reached the required prediction threshold."
                )

            else:

                for rank, prediction in enumerate(
                    predictions,
                    start=1
                ):

                    crop = prediction["crop"]
                    confidence = prediction["confidence"]

                    if rank == 1:
                        medal = "🥇"
                    elif rank == 2:
                        medal = "🥈"
                    elif rank == 3:
                        medal = "🥉"
                    else:
                        medal = "🌱"

                    st.markdown(
                        f"""
                        <div class="result-box">
                            <b style="font-size:20px;">
                                {medal} {crop.title()}
                            </b>
                            <br>
                            Prediction Probability:
                            <b>{confidence:.2f}%</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(
                        min(confidence / 100, 1.0)
                    )


            # Explanation
            st.markdown(
                '<div class="info-box">'
                '<b>🌱 About the prediction</b>'
                '<br><br>'
                'The percentage represents the model\'s prediction '
                'probability for each supported crop based on the '
                'entered soil and weather conditions.'
                '</div>',
                unsafe_allow_html=True
            )

        except Exception as e:

            st.error(
                f"Something went wrong while predicting: {e}"
            )


# Supported crops
st.markdown(
    '<div class="section-title">🌾 Supported Crops</div>',
    unsafe_allow_html=True
)


supported_crops = [
    "•Apple",
    "Banana",
    "Blackgram",
    "Chickpea",
    "Coconut",
    "Coffee",
    "Cotton",
    "Grapes",
    "Jute",
    "Kidneybeans",
    "Lentil",
    "Maize",
    "Mango",
    "Mothbeans",
    "Mungbean",
    "Muskmelon",
    "Orange",
    "Papaya",
    "Pigeonpeas",
    "Pomegranate",
    "Rice",
    "Watermelon"
]


st.markdown(
    '<div class="crops-box">'
    + " • ".join(supported_crops)
    + '</div>',
    unsafe_allow_html=True
)