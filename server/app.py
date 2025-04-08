from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from fer import FER

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load the FER model
detector = FER(mtcnn=True)

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    data = request.get_json()
    image_data = data.get('image')

    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    try:
        header, encoded = image_data.split(",")
        img_bytes = base64.b64decode(encoded)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, flags=cv2.IMREAD_COLOR)
        print("🖼️ Image decoded successfully.")

        result = detector.detect_emotions(img)
        print("🔍 Detection result length:", len(result))

        if result:
            top_emotion, score = detector.top_emotion(img)
            print(f"✅ Top emotion: {top_emotion} (score: {score})")
            return jsonify({"emotion": top_emotion})
        else:
            print("😐 No face detected. Returning 'neutral'.")
            return jsonify({"emotion": "neutral"})

    except Exception as e:
        print("🚨 Error during processing:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
