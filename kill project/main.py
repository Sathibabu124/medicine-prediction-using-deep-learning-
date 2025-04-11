# from flask import Flask, request, jsonify, render_template
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import pickle

# app = Flask(__name__)

# # Load the model, tokenizer, and encoders
# model = load_model("model/lstm_model.h5")
# tokenizer = pickle.load(open("model/tokenizer.pkl", "rb"))
# label_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))
# medicine_encoder = pickle.load(open("model/medicine_encoder.pkl", "rb"))

# # Example medicine DataFrame
# medicine_data = {
#     'Disease': ['Flu', 'Migraine', 'Diabetes'],
#     'Recommended_Medicines': [
#         ['Paracetamol', 'Cough Syrup'],
#         ['Ibuprofen', 'Sumatriptan'],
#         ['Metformin', 'Insulin']
#     ]
# }
# medicine_df = pd.DataFrame(medicine_data)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     symptoms = data['symptoms']
    
#     # Tokenize and pad the input symptoms
#     sequence = tokenizer.texts_to_sequences([symptoms])
#     padded_sequence = pad_sequences(sequence, maxlen=10, padding='post')
    
#     # Predict the disease
#     prediction = model.predict(padded_sequence)
#     predicted_label = np.argmax(prediction, axis=1)
#     disease = label_encoder.inverse_transform(predicted_label)[0]
    
#     # Get recommended medicines
#     recommended_medicines = medicine_df[medicine_df['Disease'] == disease]['Recommended_Medicines'].values[0]
    
#     return jsonify({
#         'disease': disease,
#         'recommended_medicines': recommended_medicines
#     })

# if __name__ == '__main__':
#     app.run(debug=True)