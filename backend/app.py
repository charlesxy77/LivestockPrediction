import os
import pickle
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
import numpy as np
import torch
import torch.nn as nn
import boto3

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
        s3 = boto3.client('s3',
                          aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                          aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
        bucket_name = os.environ['S3_BUCKET_NAME']
        model_key = 'LiveStock_model.pkl'
        
        response = s3.get_object(Bucket=bucket_name, Key=model_key)
        model_str = response['Body'].read()
        model = pickle.loads(model_str)
        
        logger.info("Model loaded successfully from S3")
    except Exception as e:
        logger.error(f"Failed to load the model: {str(e)}")
        raise

@app.route('/api/predict', methods=['POST'])
def predict():
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

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({"message": "Backend is running!"}), 200

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