Smart Vehical — AI Based Car Damage Detection & Repair Cost Estimator
Deep Learning • YOLOv8 • Flask Web App

Smart Vehical is an AI-powered automated car inspection system designed to analyze car damage from a single uploaded image.
The system uses Deep Learning + YOLO Object Detection + Custom Cost Algorithms to:

***Detect the type of car***

***Identify the severity of damage***

***Locate the exact damaged parts***

***Estimate the total repair cost automatically***

This project provides an end-to-end pipeline useful for insurance companies, car workshops, service centers, automobile showrooms, and smart inspection kiosks.

The goal is to build a fully automated AI solution that reduces manual inspection time, avoids human errors, and assists in cost evaluation.

*****Key Features******
🚘 1. Car Type Detection

Detects the car type from the uploaded image using a trained CNN model.
Supports common categories like:

SUV

Sedan

Hatchback

Sports Car

Luxury Cars

🩸 2. Damage Severity Classification

A deep learning model predicts how severe the damage is:

Minor Damage

Moderate Damage

Severe Damage

This helps in cost estimation and repair difficulty assessment.

🔧 3. YOLO-Based Damaged Parts Detection

Using YOLOv8, the system automatically identifies damaged parts such as:

Bumper

Door

Headlight

Fender

Bonnet

Windshield

Tyre
and more…

Bounding boxes with confidence score help clearly visualize damaged regions.

💰 4. Intelligent Repair Cost Estimation

The final repair cost is computed based on:

Car Type

Severity Level

Number & type of damaged parts

Customizable base cost logic

The estimation system is flexible and can be expanded for real-world use.

🎨 5. Modern Web Interface (Sapphire Blue Theme)

Clean & premium UI

Step-wise progress indicator

Live image preview

Interactive animations (spiral background)

Fully responsive layout

🔄 6. Complete End-to-End AI Pipeline

From upload → prediction → detection → cost estimation
Every step is automated and visually represented in the UI.

*****How to use this project******
1️⃣ Clone the repository
git clone https://github.com/Gaurav4069/SmartVehicalDamageDetector
cd SmartVehicaDamageDetector

2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Linux/Mac

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Start the server
python app.py

5️⃣ Open in browser
