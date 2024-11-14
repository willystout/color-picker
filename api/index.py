from flask import Flask, request, jsonify
import numpy as np
import joblib
app = Flask(__name__)

knn_model = joblib.load('/Users/williamstout/Dropbox/My Mac (William’s MacBook Pro (2))/Desktop/career_prep/color-picker/model/knn_model.joblib')

@app.route("/api/identify", methods=['GET'])
def classify_color():
    # Get the color parameter from the URL
    hex_color = request.args.get('color');
    if not hex_color:
        return jsonify({"error": "Invalid color format"}), 400
    try:
        # Convert the hex color to RGB
        rgb_color = np.array([hex_to_rgb(hex_color)])
        # Predict the color label using the KNN model
        predicted_label = knn_model.predict(rgb_color)[0]
        # Return the result as JSON
        return jsonify({"color": hex_color, "label": predicted_label})
    except ValueError:
        return jsonify({"error": "Invalid color format"}), 400


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4));