import cv2
import os
import base64
from flask import Flask, render_template, request
from model import image_processing
import tempfile

app = Flask(__name__)

@app.route("/")
def main():
    return render_template("index.html")

@app.route('/detection', methods=['GET', 'POST'])
def detect_image_file():
    try:
        if request.method == 'POST':
            uploaded_file = request.files['file']
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            
            uploaded_file.save(temp_file.name)
            processed_image = image_processing(temp_file.name)

            _, buffer = cv2.imencode('.jpg', processed_image)
            image_encoded = base64.b64encode(buffer).decode('utf-8')

            temp_file.close()
            os.remove(temp_file.name)

            return render_template("result.html", image=image_encoded)
        else:
            error = "Metodo invalido."
            return render_template("result.html", err=error)

    except Exception as e:
        error = "Archivo no pudo ser procesado."
        return render_template("result.html", err=error)

if __name__ == "__main__":
    app.run(port=9000, debug=True)
