# utils/cost_estimator.py
# -----------------------------------------------------------
# FINAL VERSION — Supports all 196 car names automatically
# -----------------------------------------------------------

# -----------------------------------------
# 1) BASE COST FOR PARTS
# -----------------------------------------
PART_COST = {
    "bumper": 4000,
    "door": 7000,
    "hood": 8000,
    "fender": 5000,
    "headlight": 2500,
    "taillight": 2000,
    "windshield": 9000,
    "mirror": 1500,
    "grille": 3000,
    "other": 3000
}

# -----------------------------------------
# 2) SEVERITY MULTIPLIERS
# -----------------------------------------
SEVERITY_MULTIPLIER = {
    "no_damage": 0.0,
    "minor": 1.0,
    "moderate": 1.5,
    "severe": 2.2
}

# -----------------------------------------
# 3) CAR CATEGORY COST MULTIPLIERS
# -----------------------------------------
CAR_TYPE_MULTIPLIER = {
    "hatchback": 1.0,
    "sedan": 1.2,
    "suv": 1.5,
    "luxury": 2.0,
    "unknown": 1.0
}

# -----------------------------------------
# 4) AUTO-DETECT CAR CATEGORY FROM NAME
# -----------------------------------------
def detect_car_category(car_name: str):
    car = car_name.lower()

    # SUVs
    if "suv" in car:
        return "suv"
    
    # Hatchbacks
    if "hatchback" in car:
        return "hatchback"
    
    # Sedans
    if "sedan" in car:
        return "sedan"
    
    # Coupe → treat as sedan
    if "coupe" in car:
        return "sedan"
    
    # Convertible → treat as luxury
    if "convertible" in car:
        return "luxury"
    
    # Wagon → treat as hatchback
    if "wagon" in car:
        return "hatchback"
    
    # Van / Minivan → treat as hatchback
    if "van" in car or "minivan" in car:
        return "hatchback"
    
    # Pickup / Crew Cab → treat as SUV
    if "pickup" in car or "crew cab" in car:
        return "suv"

    return "unknown"


# -----------------------------------------
# 5) NORMALIZE YOLO PART NAMES
# -----------------------------------------
def normalize(part: str):
    p = part.lower()

    if "bumper" in p:
        return "bumper"
    if "door" in p:
        return "door"
    if "hood" in p or "bonnet" in p:
        return "hood"
    if "fender" in p:
        return "fender"
    if "head" in p and "light" in p:
        return "headlight"
    if "tail" in p and "light" in p:
        return "taillight"
    if "windshield" in p or "windscreen" in p:
        return "windshield"
    if "mirror" in p:
        return "mirror"
    if "grill" in p or "grille" in p:
        return "grille"

    return "other"


# -----------------------------------------
# 6) FINAL COST ESTIMATOR
# -----------------------------------------
def estimate_repair_cost(car_type_name: str, severity: str, damaged_parts: dict):

    # STEP 1 — Auto car category detection
    category = detect_car_category(car_type_name)

    # STEP 2 — Base cost for all damaged parts
    base_sum = 0
    for part, count in damaged_parts.items():
        norm = normalize(part)
        base_sum += PART_COST.get(norm, 3000) * count

    # STEP 3 — Apply severity effect
    severity_mult = SEVERITY_MULTIPLIER.get(severity.lower(), 1.0)

    # STEP 4 — Apply car-type multiplier
    car_mult = CAR_TYPE_MULTIPLIER.get(category, 1.0)

    # STEP 5 — Final repair cost
    final_cost = base_sum * severity_mult * car_mult

    # STEP 6 — Round to nearest ₹100
    return int(round(final_cost, -2))
