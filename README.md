# 🌱 Smart Crop Recommendation System

A machine learning-based web application that recommends suitable crops based on soil and weather conditions.

The system takes seven environmental and soil parameters as input and uses a Multi-Layer Perceptron (MLP) neural network to predict the most suitable crops.

## 🚀 Features

* 🌱 Crop recommendation based on soil and weather conditions
* 🤖 Multi-Layer Perceptron (MLP) neural network
* 📊 Prediction probabilities for recommended crops
* 🌾 Supports 22 different crop classes
* ⚡ FastAPI backend for model inference
* 🖥️ Streamlit frontend
* 📱 Simple and mobile-friendly interface
* 🔢 Input validation for reasonable values

## 🧠 Model

The model is trained using the **Crop Recommendation Dataset**, which contains 2,200 samples and 7 input features.

### Input Features

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* Soil pH
* Rainfall

### Model Architecture

The prediction model is a Multi-Layer Perceptron built using TensorFlow/Keras.

```text
Input Layer
    ↓
Dense Layer (128 neurons, ReLU)
    ↓
Dropout (20%)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Dropout (20%)
    ↓
Output Layer (22 neurons, Softmax)
```

### Model Performance

* **Training Accuracy:** 98.90%
* **Test Accuracy:** 98.79%
* **Best Validation Accuracy:** 99.70%

## 🌾 Supported Crops

The model currently supports the following 22 crops:

* Apple
* Banana
* Blackgram
* Chickpea
* Coconut
* Coffee
* Cotton
* Grapes
* Jute
* Kidneybeans
* Lentil
* Maize
* Mango
* Mothbeans
* Mungbean
* Muskmelon
* Orange
* Papaya
* Pigeonpeas
* Pomegranate
* Rice
* Watermelon

> **Note:** The model can only recommend crops that were included in its training classes.

## 🏗️ Project Structure

```text
smart-crop-recommendation-system/
│
├── back_end/
│   ├── main.py
│   ├── best_crop_mlp.keras
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── front_end/
│   └── app.py
│
├── requirements.txt
└── README.md
```

## ⚙️ Technologies Used

* Python
* TensorFlow
* Keras
* Scikit-learn
* Pandas
* NumPy
* FastAPI
* Uvicorn
* Streamlit
* Joblib

## 🔄 How It Works

```text
User Input
    ↓
Streamlit Frontend
    ↓
FastAPI Backend
    ↓
Data Scaling
    ↓
MLP Model
    ↓
Prediction Probabilities
    ↓
Recommended Crops
```

The user enters the soil and weather conditions through the Streamlit interface.

The frontend sends the data to the FastAPI backend, where the input is scaled using the same scaler used during training.

The trained MLP model then predicts the probability of each supported crop.

The results are sorted by prediction probability and displayed to the user.

## 🖥️ Running the Project

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

```bash
cd smart-crop-recommendation-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI Backend

Open a terminal and navigate to:

```bash
cd back_end
```

Then run:

```bash
uvicorn main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

FastAPI documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### 4. Run the Streamlit Frontend

Open another terminal and navigate to:

```bash
cd front_end
```

Then run:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

## 🔌 API Endpoint

### POST `/predict`

The API accepts the following parameters:

```json
{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 25.5,
    "humidity": 80,
    "ph": 6.5,
    "rainfall": 200
}
```

Example response:

```json
{
    "predictions": [
        {
            "crop": "jute",
            "confidence": 61.09
        },
        {
            "crop": "rice",
            "confidence": 38.87
        }
    ]
}
```

## 📌 Important Note

The percentages shown by the application represent the model's **prediction probability/confidence** for each supported crop.

They should not be interpreted as a guaranteed probability of successful farming or crop yield.

The model is trained only on the supported crop classes listed above, so it cannot recommend crops outside those classes.

## 📊 Dataset

The project uses the **Crop Recommendation Dataset**, containing 2,200 samples across 22 crop classes.

Each crop class contains 100 samples.

## 👩‍💻 Project

**Smart Crop Recommendation System**

Built using Machine Learning, FastAPI, and Streamlit.
