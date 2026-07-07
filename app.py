from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import joblib
import base64
import re
from google import genai

app = Flask(__name__)

# =======================
# GEMINI CONFIG
# =======================

client = genai.Client(
    api_key="AQ.Ab8RN6I2AIUJj6zCeaY2Rr7-9jzZfS0X42hp3Ahhyp2EMCIrQQ"
)


# =======================
# SENTENCE GENERATION
# =======================
def call_gemini(sequence):
    try:
        text = " ".join(sequence).lower().strip()

        prompt = f"""
You are an expert sign language translator.

Convert word sequence into a natural English sentence.

RULES:
- Fix grammar
- Fix word order
- Add missing words
- Output ONLY one sentence

Examples:
Input: hi how are you
Output: Hi, how are you?

Input: i fine
Output: I am fine.

Input: i go market tomorrow
Output: I will go to the market tomorrow.

Now convert:
Input: {text}
Output:
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text.strip()

    except Exception as e:
        print("Gemini Error:", e)
        return " ".join(sequence)


# =======================
# INTENT DETECTION
def detect_intent(sentence):
    try:
        prompt = f"""
You are a STRICT intent classification system.

You MUST classify the sentence into EXACTLY ONE category from below:

CATEGORIES:
1. Greeting → hi, hello, how are you
2. Emergency → help, danger, call ambulance, save me
3. Medical → I am sick, I need water, I am hungry, pain
4. Food → eat, hungry, food, restaurant
5. Travel → go, come, airport, bus, train, market
6. Question → what, where, how, when, why
7. General → everything else

RULES:
- Output ONLY the category name (no explanation)
- Do NOT output anything else
- Be strict and accurate

EXAMPLES:

Sentence: hi
Output: Greeting

Sentence: help me
Output: Emergency

Sentence: I need water
Output: Medical

Sentence: I want food
Output: Food

Sentence: where are you going
Output: Question

NOW CLASSIFY:

Sentence: {sentence}
Output:
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        result = response.text.strip()

        # extra safety cleanup
        valid = [
            "Greeting",
            "Emergency",
            "Medical",
            "Food",
            "Travel",
            "Question",
            "General"
        ]

        for v in valid:
            if v.lower() in result.lower():
                return v

        return "General"

    except Exception as e:
        print("Intent Error:", e)
        return "General"

# =======================
# LOAD MODEL
# =======================
model = joblib.load("gesture_model.pkl")
encoder = joblib.load("label_encoder.pkl")
expected_features = model.n_features_in_

# =======================
# MEDIA PIPE
# =======================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =======================
# STATE
# =======================
gesture_buffer = []
last_gesture = None
repeat_count = 0
THRESHOLD = 3


# =======================
# ROUTES
# =======================
@app.route('/')
def home():
    return render_template("index.html")


@app.route('/capture')
def capture():
    return render_template("capture.html")


@app.route('/reset', methods=['POST'])
def reset():
    global gesture_buffer, last_gesture, repeat_count
    gesture_buffer = []
    last_gesture = None
    repeat_count = 0
    return jsonify({"status": "reset done"})


# =======================
# PREDICT
# =======================
@app.route('/predict', methods=['POST'])
def predict():
    global last_gesture, repeat_count, gesture_buffer

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"gesture": "invalid input", "buffer": gesture_buffer})

    img_data = data["image"]
    match = re.search(r'base64,(.*)', img_data)
    if not match:
        return jsonify({"gesture": "invalid image", "buffer": gesture_buffer})

    img_bytes = base64.b64decode(match.group(1))
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"gesture": "frame error", "buffer": gesture_buffer})

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        last_gesture = None
        repeat_count = 0
        return jsonify({"gesture": "No hand detected", "buffer": gesture_buffer})

    hand_landmarks = result.multi_hand_landmarks[0]

    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])

    features = features[:expected_features]
    features += [0] * (expected_features - len(features))

    prediction = model.predict([features])
    gesture = encoder.inverse_transform(prediction)[0]

    # stability filter
    if gesture == last_gesture:
        repeat_count += 1
    else:
        repeat_count = 0
        last_gesture = gesture

    if repeat_count >= THRESHOLD:
        if not gesture_buffer or gesture_buffer[-1] != gesture:
            gesture_buffer.append(gesture)

    return jsonify({
        "gesture": gesture,
        "buffer": gesture_buffer
    })


# =======================
# FINISH (GEMINI + INTENT)
# =======================
@app.route('/finish', methods=['POST'])
def finish():
    global gesture_buffer, last_gesture, repeat_count

    if not gesture_buffer:
        return jsonify({
            "final_sentence": "No gesture detected",
            "intent": "None"
        })

    # sentence generation
    sentence = call_gemini(gesture_buffer.copy())

    # intent detection
    intent = detect_intent(sentence)

    # reset state
    gesture_buffer = []
    last_gesture = None
    repeat_count = 0

    return jsonify({
        "final_sentence": sentence,
        "intent": intent
    })


# =======================
# RUN
# =======================
if __name__ == "__main__":
    print("Server running at http://127.0.0.1:5000/")
    app.run(host="127.0.0.1", port=5000, debug=True)