import os
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# -------------------------
# Config
# -------------------------
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
MODEL_PATH = r"C:\Users\Pc\OneDrive\Desktop\driver drowsness\DDD_split\best_model.h5"
model = load_model(MODEL_PATH)

# Check input shape
INPUT_SIZE = model.input_shape[1:3]  # e.g. (224, 224)


# -------------------------
# Helper Functions
# -------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    if pred.shape[1] == 1:  # sigmoid
        prob = float(pred[0][0])
        if prob < 0.5:
            return f"Drowsy (Probability: {prob:.4f})"
        else:
            return f"Not Drowsy (Probability: {prob:.4f})"
    else:  # softmax
        class_idx = np.argmax(pred, axis=1)[0]
        prob = pred[0][class_idx]
        if class_idx == 1:
            return f"Drowsy (Probability: {prob:.4f})"
        else:
            return f"Not Drowsy (Probability: {prob:.4f})"


# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            result = predict_image(file_path)

            return render_template("index.html", filename=filename, result=result)

    return render_template("index.html")


@app.route("/display/<filename>")
def display_image(filename):
    return redirect(url_for("static", filename="uploads/" + filename), code=301)


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
