from flask import Flask,render_template,request
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os


# 
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def preprocess_image(imagepath):
    image=cv2.imread(imagepath)
    image_resize= cv2.resize(image,(120,120))
    image_scaled = image_resize/255
    image_reshape=np.reshape(image_scaled,[1,120,120,3])
    return image_reshape



@app.route("/")
def index():
    return render_template("homepage.html")

model = tf.keras.models.load_model("C:\\Users\\LENOVO-PC\\Videos\\Projects\\Deep.learing.facemask\\FMMM.h5")

@app.route("/predict", methods=["POST", "GET"])
def upload():
    if request.method == "POST":
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            processed_image = preprocess_image(file_path)


            prediction = model.predict(processed_image)
            image_answer = np.argmax(prediction)

            with_mask = "with mask"  
            without_mask = "without mask"  

            return render_template("homepage.html", image_answer=image_answer, withoutMask=without_mask, withMask=with_mask)
    
    return render_template("homepage.html")

if __name__ == "__main__":
    app.run(debug=True)
