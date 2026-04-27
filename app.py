from flask import Flask, request, render_template_string
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
model = load_model("cnn_pneumonia_model.h5")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Pneumonia Detection</title>
    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #dfe9f3, #ffffff);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .card {
            background: white;
            width: 520px;
            padding: 35px;
            border-radius: 22px;
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
            text-align: center;
        }

        h1 {
            margin-bottom: 8px;
            color: #1f2937;
        }

        .subtitle {
            color: #6b7280;
            margin-bottom: 30px;
        }

        input[type="file"] {
            margin: 15px 0;
        }

        button {
            background: #2563eb;
            color: white;
            border: none;
            padding: 12px 22px;
            border-radius: 12px;
            cursor: pointer;
            font-size: 16px;
            transition: 0.2s;
        }

        button:hover {
            background: #1d4ed8;
            transform: translateY(-2px);
        }

        img {
            max-width: 100%;
            margin-top: 25px;
            border-radius: 16px;
            border: 1px solid #e5e7eb;
        }

        .result {
            margin-top: 25px;
            font-size: 24px;
            font-weight: bold;
            padding: 14px;
            border-radius: 14px;
            background: #eff6ff;
            color: #1e40af;
        }

        .footer {
            margin-top: 25px;
            color: #9ca3af;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>🩺 Pneumonia Detection</h1>
        <p class="subtitle">Upload a chest X-ray image to predict NORMAL or PNEUMONIA</p>

        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <br>
            <button type="submit">Analyze X-ray</button>
        </form>

        {% if img_path %}
            <img src="{{ img_path }}">
            <div class="result">{{ result }}</div>
        {% endif %}

        <div class="footer">
            CNN model prediction demo
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    img_path = None

    if request.method == "POST":
        file = request.files["file"]

        os.makedirs("static/uploads", exist_ok=True)
        img_path = os.path.join("static/uploads", file.filename)
        file.save(img_path)

        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0][0]

        if pred > 0.5:
            result = f"PNEUMONIA — confidence: {pred:.2%}"
        else:
            result = f"NORMAL — confidence: {(1 - pred):.2%}"

        img_path = "/" + img_path.replace("\\", "/")

    return render_template_string(HTML, result=result, img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)