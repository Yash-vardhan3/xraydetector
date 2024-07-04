from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import pickle

app = Flask(__name__)

# Load the trained model from .h5 file (if needed)
model_h5 = tf.keras.models.load_model('healthcare_diagnostics_model.h5')

# Load the trained model from .pkl file
with open('healthcare_diagnostics_model.pkl', 'rb') as f:
    model_pkl = pickle.load(f)

# Choose which model to use for prediction
model = model_pkl  # Change to model_h5 if you want to use the .h5 model instead

def predict_image(image_path, model):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read the image file {image_path}. Please check the file path or the file integrity.")
    
    image = cv2.resize(image, (150, 150))
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = model.predict(image)
    return 'Pneumonia' if prediction[0][0] > 0.5 else 'Normal'

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = f"uploads/{file.filename}"
    file.save(file_path)

    try:
        prediction = predict_image(file_path, model)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
