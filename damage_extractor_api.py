import requests
import cv2
import random

API_KEY = "gL5QCD8pKajC4v5pr3X3"
MODEL_ID = "car-damage-detection-t0g92/3"
API_URL = f"https://detect.roboflow.com/{MODEL_ID}"


class DamageExtractorAPI:

    def __init__(self):
        pass

    # ----------------------------------------------------
    # API Request to Roboflow
    # ----------------------------------------------------
    def infer_api(self, image_path):
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = requests.post(
            API_URL,
            params={"api_key": API_KEY},
            files={"file": image_bytes}
        )
        return response.json()

    # ----------------------------------------------------
    # Extract prediction info + severity + damaged parts
    # ----------------------------------------------------
    def extract(self, image_path):
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        img_area = h * w if h and w else 1

        result = self.infer_api(image_path)
        preds = result.get("predictions", [])

        damaged_parts = {}
        total_area = 0

        for pred in preds:
            part = pred["class"]           # "Door", "Bumper", etc.
            width = pred["width"]
            height = pred["height"]

            box_area = width * height
            total_area += box_area

            if part not in damaged_parts:
                damaged_parts[part] = 0
            damaged_parts[part] += 1

        damage_ratio = total_area / img_area

        if len(preds) == 0:
            severity = "no_damage"
        elif damage_ratio < 0.02:
            severity = "minor"
        elif damage_ratio < 0.08:
            severity = "moderate"
        else:
            severity = "severe"

        return {
            "severity": severity,
            "damage_ratio": damage_ratio,
            "damaged_parts": damaged_parts,
            "num_damaged_parts": len(damaged_parts),
            "raw_predictions": preds
        }

    # ----------------------------------------------------
    # Bounding Box Visualization
    # ----------------------------------------------------
    def visualize(self, image_path, predictions, save_path="static/output.jpg"):
        img = cv2.imread(image_path)

        if img is None:
            print("❌ Could not load image.")
            return

        YELLOW = (0, 255, 255)
        TEXT_COLOR = (0, 0, 0)

        for pred in predictions:
            x = pred["x"]
            y = pred["y"]
            w = pred["width"]
            h = pred["height"]
            cls = pred["class"]
            conf = pred["confidence"]

            # YOLO center format → corner coords
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # Draw yellow bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), YELLOW, 3)

            label = f"{cls} ({conf:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thick = 2

            # Text size (to draw background)
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thick)

            # Label background
            cv2.rectangle(
                img,
                (x1, y1 - th - 10),
                (x1 + tw + 10, y1),
                YELLOW,
                cv2.FILLED
            )

            # Black label text
            cv2.putText(
                img,
                label,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                TEXT_COLOR,
                thick
            )

        # ----------------------------
        # SAVE ONLY — NO POPUP WINDOW
        # ----------------------------
        cv2.imwrite(save_path, img)

        print("✔ Saved visualization to:", save_path)

