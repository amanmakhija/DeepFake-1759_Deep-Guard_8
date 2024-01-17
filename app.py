from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

app = Flask(__name__)

# Load the pre-trained model
model = load_model('xception_deepfake_image.h5')

# Load the Haarcascades face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            # Read and preprocess the image
            img_array = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

            # Ensure the image is successfully decoded
            if img is None:
                return jsonify({'error': 'Failed to decode image'})

            # If the image has an alpha channel, remove it
            if img.shape[-1] == 4:
                img = img[:, :, :3]

            # Ensure the image has 3 channels (for compatibility with preprocess_input)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.shape[-1] == 1 else img

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                return jsonify({'error': 'No face detected'})

            # Crop the first detected face
            x, y, w, h = faces[0]
            cropped_face = img[y:y+h, x:x+w]

            # Resize the cropped face to match the input size expected by the model
            cropped_face = cv2.resize(cropped_face, (224, 224))
            cropped_face = preprocess_input(cropped_face.reshape(1, 224, 224, 3))

            # Make prediction
            prediction = model.predict(cropped_face)

            # Convert the prediction to a human-readable label
            label = 'FAKE' if prediction[0][0] > 0.5 else 'REAL'

            return jsonify({'prediction': label})
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)