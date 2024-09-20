import os
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import numpy as np
import torch
import torch.nn as nn

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiOutputModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiOutputModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

model = None

def load_model():
    global model
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'LiveStock_model.pkl')
        logger.info(f"Looking for model file at: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load the model: {str(e)}")
        raise

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json
        logger.info(f"Received data: {data}")
        
        # Convert input data to a PyTorch tensor
        input_data = torch.tensor([float(data[key]) for key in ['DM', 'CP', 'CF', 'NDF', 'ADF', 'ADL', 'ASH']], dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_data).numpy()
        
        # Convert prediction to dictionary and round to 2 decimal places
        result = {
            "DMD": round(prediction[0].item(), 2),
            "OMD": round(prediction[1].item(), 2),
            "ME": round(prediction[2].item(), 2),
            "CH4": round(prediction[3].item(), 2)
        }
        
        logger.info(f"Prediction result: {result}")
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Backend is running!"}), 200

if __name__ == '__main__':
    try:
        load_model()
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=False, host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"Could not start the app. Error: {str(e)}")