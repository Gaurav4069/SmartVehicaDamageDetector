import os
import json
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

from utils.preprocess import preprocess_image
from utils.cost_estimator import estimate_repair_cost

from damage_extractor_api import DamageExtractorAPI

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load ML Models
car_model = tf.keras.models.load_model("model/model/car_model_detector.keras")
severity_model = tf.keras.models.load_model("model/final_vehicle_damage_model.keras")

with open("model/car_labels.json", "r") as f:
    car_labels = json.load(f)

extractor = DamageExtractorAPI()

# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")
 

# ---------------------------------------------------------
# STEP 1: CAR TYPE DETECTION
# ---------------------------------------------------------
@app.route("/detect_car")
def detect_car():
    return render_template("detect_car.html", step=1)


@app.route("/predict_car", methods=["POST"])
def predict_car():
    file = request.files["image"]
    path = os.path.join(UPLOAD_FOLDER, "car_input.jpg")
    file.save(path)

    img = preprocess_image(path)
    pred = car_model.predict(np.array([img]))[0]
    index = np.argmax(pred)
    car_type = car_labels[str(index)]

    return render_template(
        "detect_car.html",
        image_path=path,
        car_type=car_type,
        step=1
    )


# ---------------------------------------------------------
# STEP 2: SEVERITY PREDICTION
# ---------------------------------------------------------
@app.route("/severity")
def severity():
    return render_template("severity.html", step=2)


@app.route("/predict_severity", methods=["POST"])
def predict_severity():
    file = request.files["image"]
    path = os.path.join(UPLOAD_FOLDER, "severity_input.jpg")
    file.save(path)

    img = preprocess_image(path)
    pred = severity_model.predict(np.array([img]))[0]

    labels = ["minor", "moderate", "severe"]
    severity = labels[np.argmax(pred)]

    return render_template(
        "severity.html",
        image_path=path,
        severity=severity,
        step=2
    )


# ---------------------------------------------------------
# STEP 3: YOLO DAMAGE DETECTION + COST ESTIMATION
# ---------------------------------------------------------
@app.route("/yolo")
def yolo():
    return render_template("yolo.html", step=3)


@app.route("/predict_yolo", methods=["POST"])
def predict_yolo():
    file = request.files["image"]
    img_path = os.path.join(UPLOAD_FOLDER, "yolo_input.jpg")
    file.save(img_path)

    result = extractor.extract(img_path)
    preds = result["raw_predictions"]

    output_path = "static/output.jpg"
    extractor.visualize(img_path, preds, output_path)

    car_type = request.form.get("car_type", "sedan")
    severity = result["severity"]
    damaged_parts = result["damaged_parts"]

    import json

    return render_template(
        "yolo.html",
        image_path=output_path,
        severity=severity,
        damaged_parts=damaged_parts,
        damaged_parts_json=json.dumps(damaged_parts),
        car_type=car_type,
        step=3
    )

 
@app.route("/cost")
def cost_page():
    car_type = request.args.get("car_type")
    severity = request.args.get("severity")
    damaged_parts = json.loads(request.args.get("damaged_parts"))

    estimated_cost = estimate_repair_cost(
        car_type,
        severity,
        damaged_parts
    )

    return render_template(
        "cost.html",
        car_type=car_type,
        severity=severity,
        damaged_parts=damaged_parts,
        estimated_cost=estimated_cost,
        step=4
    )



# ---------------------------------------------------------
# RUN SERVER
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
