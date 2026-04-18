from flask import Flask, render_template, request, redirect, url_for, flash
import os
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from utils import predict_age

# =========================
# App Config
# =========================
app = Flask(__name__)
app.secret_key = "secret_key"  # required for flash messages

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# Load Model
# =========================
model = load_model("models/best_model1.keras")


# =========================
# Helper Functions
# =========================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# =========================
# Routes
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        # Check if file exists
        if "image" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["image"]

        # If user submits empty form
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        # Validate file
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            try:
                prediction, confidence = predict_age(model, filepath)
                image_path = filepath
            except Exception as e:
                flash(f"Error during prediction: {str(e)}")
                return redirect(request.url)
        else:
            flash("Invalid file type. Only jpg, jpeg, png allowed.")
            return redirect(request.url)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )


# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=True)