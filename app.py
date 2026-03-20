from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow frontend to access

# ✅ Load trained ML model
model = pickle.load(open("best_model.pkl", "rb"))

# In-memory store for nodes
nodes_list = []

# ------------------- ROOT -------------------
@app.route("/")
def home():
    return "✅ Backend is running!"

# ------------------- PREDICTION -------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        distance = float(data.get("distance", 0))
        diff = float(data.get("diff", 0))
        slope = float(data.get("slope", 0))

        features = np.array([[distance, diff, slope]])
        prediction = model.predict(features)

        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

# ------------------- SENSOR DATA -------------------
@app.route("/sensor-data", methods=["GET"])
def get_sensor_data():
    # Dummy sensor data
    data = [
        {"distance": 80, "temperature": 25, "time": "10:00"},
        {"distance": 75, "temperature": 26, "time": "10:10"},
        {"distance": 70, "temperature": 27, "time": "10:20"},
    ]
    return jsonify(data)

# ------------------- NODES -------------------
@app.route("/nodes", methods=["GET"])
def get_nodes():
    # Return current nodes list
    return jsonify(nodes_list)

@app.route("/nodes", methods=["POST"])
def create_node():
    try:
        data = request.json
        node = {
            "id": data.get("id"),
            "height": data.get("height"),
            "length": data.get("length"),
            "width": data.get("width"),
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude")
        }
        nodes_list.append(node)
        return jsonify({"message": "Node created", "node": node})
    except Exception as e:
        return jsonify({"error": str(e)})

# ------------------- RUN -------------------
if __name__ == "__main__":
    app.run(debug=True)