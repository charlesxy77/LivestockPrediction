import os
import pickle
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
import numpy as np
import torch
import torch.nn as nn

app = Flask(__name__, static_folder='../frontend/build')
CORS(app)

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
        # List of possible locations for the model file
        possible_locations = [
            os.path.join(os.getcwd(), 'LiveStock_model.pkl'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LiveStock_model.pkl'),
            '/app/LiveStock_model.pkl',  
        ]

        for model_path in possible_locations:
            logger.info(f"Trying to load model from: {model_path}")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                model.to('cpu')
                logger.info(f"Model loaded successfully from {model_path}")
                return

        raise FileNotFoundError("Model file not found in any of the expected locations")

    except Exception as e:
        logger.error(f"Failed to load the model: {str(e)}")
        raise


load_model()
    

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json
        logger.info(f"Received data: {data}")
        
        # Convert input data to a PyTorch tensor
        input_data = torch.tensor([float(data[key]) for key in ['DM', 'CP', 'CF', 'NDF', 'ADF', 'ADL', 'ASH']], dtype=torch.float32).to('cpu')
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_data).cpu().numpy()
        
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

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({"message": "Backend is running!"}), 200


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"message": "Success!"}), 200

# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    try:
        load_model()
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=False, host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"Could not start the app. Error: {str(e)}")