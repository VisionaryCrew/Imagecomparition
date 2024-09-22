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

    # Save original images
    original_image1_path = 'static/original_image1.png'
    original_image2_path = 'static/original_image2.png'
    if not os.path.exists('static'):
        os.makedirs('static')
    cv2.imwrite(original_image1_path, img1)
    cv2.imwrite(original_image2_path, img2)

    # Resize img2 to the size of img1 if necessary
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Compute the absolute differences
    diff = cv2.absdiff(img1, img2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

    # Create masks for highlighting differences
    mask1 = np.zeros_like(img1)
    mask2 = np.zeros_like(img2)
    differences_image = np.zeros_like(img1)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter out small contours
            # Highlight in the masks
            cv2.drawContours(mask1, [contour], -1, (0, 0, 255), thickness=cv2.FILLED)  # Red for img1
            cv2.drawContours(mask2, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)  # Green for img2
            
            # Add to differences image
            cv2.drawContours(differences_image, [contour], -1, (0, 255, 255), thickness=cv2.FILLED)  # Yellow for both

    # Highlight the differences in both images
    highlighted_image1 = cv2.addWeighted(img1, 1, mask1, 0.5, 0)
    highlighted_image2 = cv2.addWeighted(img2, 1, mask2, 0.5, 0)

    # Create final differences image
    differences_image[(mask1[:,:,0] == 255) & (mask2[:,:,0] == 0)] = [0, 0, 255]  # Red for unique in img1
    differences_image[(mask2[:,:,0] == 255) & (mask1[:,:,0] == 0)] = [0, 255, 0]  # Green for unique in img2
    differences_image[(mask1[:,:,0] == 255) & (mask2[:,:,0] == 255)] = [0, 255, 255]  # Yellow for common differences

    # Save the result images
    highlighted_image1_path = 'static/highlighted_image1.png'
    highlighted_image2_path = 'static/highlighted_image2.png'
    differences_image_path = 'static/differences_image.png'

    cv2.imwrite(highlighted_image1_path, highlighted_image1)
    cv2.imwrite(highlighted_image2_path, highlighted_image2)
    cv2.imwrite(differences_image_path, differences_image)

    return jsonify(
        originalImage1=original_image1_path,
        originalImage2=original_image2_path,
        highlightedImage1=highlighted_image1_path,
        highlightedImage2=highlighted_image2_path,
        differencesImage=differences_image_path
    )

if __name__ == '__main__':
    app.run(debug=True)
