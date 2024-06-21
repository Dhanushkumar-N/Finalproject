from flask import Flask, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename
import joblib
import cv2
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
model_path = 'D:/Project/models/best_model.pkl'
model = joblib.load(model_path)

# Load the label encoder
label_encoder = joblib.load('D:/Project/models/label_encoder.pkl')

# Load the dataset
dataset_path = "D:/Project/dataset.xlsx"
dataset = pd.read_excel(dataset_path)

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'D:/Project/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the uploaded image
def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (100, 100))  # Resize image to match training size
    image = image.flatten().reshape(1, -1)  # Flatten and reshape for prediction
    return image

# Function to predict plant information
def predict_plant(image):
    processed_image = process_image(image)
    predicted_class = model.predict(processed_image)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        
        file = request.files['file']
        
        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        
        # If file is uploaded and allowed, process it
        if file and allowed_file(file.filename):
            try:
                # Save the uploaded image to the uploads folder
                filename = secure_filename(file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(image_path)
                
                # Read the uploaded image
                image = cv2.imread(image_path)
                
                # Predict the class of the image
                predicted_label = predict_plant(image)
                
                # Retrieve additional information from the dataset
                plant_info = dataset[dataset['Scientific_name'] == predicted_label]
                
                if not plant_info.empty:
                    plant_info = plant_info[['Tamil_name', 'Parts_used', 'Uses', 'Grown_Area', 'Preparation_method','Preparation_method(Tamil)']].iloc[0].to_dict()
                    return render_template('index.html', message='Prediction: {}'.format(predicted_label), info=plant_info, image_path=image_path)
                else:
                    return render_template('index.html', message='No information found for the predicted plant')
            except Exception as e:
                return render_template('index.html', message='Error processing image: {}'.format(str(e)))
        
        else:
            return render_template('index.html', message='File type not allowed')
    
    return render_template('index.html', message='Upload an image')

# Route to display the uploaded image
@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
