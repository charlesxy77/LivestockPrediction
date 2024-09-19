import os
import torch
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from multi_output_model import MultiOutputModel

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None

def load_model():
    global model
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Current directory: {current_dir}")
        pth_path = os.path.join(current_dir, 'LiveStock_model.pth')
        logger.info(f"Looking for .pth file at: {pth_path}")
        pkl_path = os.path.join(current_dir, 'LiveStock_model.pkl')
        logger.info(f"Looking for .pkl file at: {pkl_path}")
        
        if os.path.exists(pth_path):
            logger.info("Found .pth file. Loading PyTorch model.")
            input_size = 7
            hidden_size = 64
            output_size = 4
            model = MultiOutputModel(input_size, hidden_size, output_size)
            model.load_state_dict(torch.load(pth_path))
            model.eval()
        elif os.path.exists(pkl_path):
            logger.info("Found .pkl file. Loading pickled model.")
            with open(pkl_path, 'rb') as f:
                model = pickle.load(f)
        else:
            raise FileNotFoundError(f"No model file found at {pth_path} or {pkl_path}")
        
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
        
        # Convert input data to a format suitable for your model
        input_data = torch.tensor([float(data[key]) for key in ['DM', 'CP', 'CF', 'NDF', 'ADF', 'ADL', 'ASH']], dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_data)
        
        # Convert prediction to list and round to 2 decimal places
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
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Could not start the app. Error: {str(e)}")