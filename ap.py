from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return send_from_directory('frontend', 'index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify(error='No images uploaded'), 400

    image1 = request.files['image1']
    image2 = request.files['image2']

    # Read images
    img1 = cv2.imdecode(np.frombuffer(image1.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(image2.read(), np.uint8), cv2.IMREAD_COLOR)

    # Check if images are loaded
    if img1 is None or img2 is None:
        return jsonify(error='Error loading images'), 400

    # Resize img2 to the size of img1 if necessary
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference using NumPy
    diff = np.abs(gray1.astype(np.int16) - gray2.astype(np.int16)).astype(np.uint8)

    # Threshold the difference to create a binary mask
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Create a colored highlight for differences
    highlighted_diff = np.zeros_like(img1)  # Create a black image with the same shape as img1
    highlighted_diff[mask == 255] = [0, 0, 255]  # Set the highlight color to red (BGR format)

    # Combine the original image with the highlighted differences
    result_image = cv2.addWeighted(img1, 0.7, highlighted_diff, 0.3, 0)

    # Save the result image
    result_image_path = 'static/result.png'
    if not os.path.exists('static'):
        os.makedirs('static')
    success = cv2.imwrite(result_image_path, result_image)
    if not success:
        return jsonify(error='Error saving the result image'), 500

    return jsonify(resultImage='/static/result.png')

if __name__ == '__main__':
    app.run(debug=True)
