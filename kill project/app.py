from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load Model & Preprocessing Objects
model = tf.keras.models.load_model("model/lstm_model.h5")
tokenizer = pickle.load(open("model/tokenizer.pkl", "rb"))
label_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))
medicine_encoder = pickle.load(open("model/medicine_encoder.pkl", "rb"))

def predict_disease_and_medicine(symptoms_input, age_input):
    symptoms_seq = tokenizer.texts_to_sequences([symptoms_input])
    symptoms_padded = pad_sequences(symptoms_seq, maxlen=10, padding='post')
    age_array = np.array([[age_input]])

    predictions = model.predict([symptoms_padded, age_array])

    # Predict Disease
    disease_pred = np.argmax(predictions[0])
    predicted_disease = label_encoder.inverse_transform([disease_pred])[0]

    # Predict Medicines (Lower threshold to 0.3)
    medicine_pred = (predictions[1] > 0.3).astype(int)
    recommended_medicines = medicine_encoder.inverse_transform(medicine_pred)[0]

    return predicted_disease, recommended_medicines

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form['symptoms']
    age = int(request.form['age'])
    
    disease, medicines = predict_disease_and_medicine(symptoms, age)
    
    return render_template('result.html', disease=disease, medicines=medicines)

if __name__ == '__main__':
    app.run(debug=True)
