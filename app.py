import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd

# Load the CSV data
disease_info = pd.read_csv(r"C:\Plant-Disease-Detection\disease_info.csv", encoding='cp1252')
supplement_info = pd.read_csv(r"C:\Plant-Disease-Detection\supplement_info.csv", encoding='cp1252')
# disease_info = pd.read_csv(r"D:\Final Project(Plant Disease)\Plant-Disease-Detection\Plant-Disease-Detection\disease_info.csv", encoding='cp1252')
# supplement_info = pd.read_csv(r"D:\Final Project(Plant Disease)\Plant-Disease-Detection\Plant-Disease-Detection\supplement_info.csv",encoding='cp1252')

# Initialize the model
model = CNN.CNN(39)  # Make sure the CNN model has 39 classes
model.load_state_dict(torch.load(r"C:\Plant-Disease-Detection\plant_disease_model_1_latest.pt"))
# model.load_state_dict(torch.load(r"D:\Final Project(Plant Disease)\Plant-Disease-Detection\Plant-Disease-Detection\plant_disease_model_1_latest.pt"))
model.eval()

# Prediction function
def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Resize image to match model input
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))  # Add batch dimension
    output = model(input_data)
    output = output.detach().numpy()  # Convert tensor to numpy array
    index = np.argmax(output)  # Get the index of the predicted class
    return index

# Initialize Flask app
app = Flask(__name__)

# Route for the homepage
@app.route('/')
def home_page():
    return render_template('home.html')

# Route for the AI engine page where users can upload an image
@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

# Route for handling image submission, prediction, and displaying results
@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        # Get the uploaded image
        image = request.files['image']
        
        # Ensure the uploads directory exists
        upload_folder = os.path.join(app.static_folder, 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        # Save the image
        filename = image.filename
        file_path = os.path.join(upload_folder, filename)
        image.save(file_path)
        
        # Print the file path for debugging
        print(f"File saved to: {file_path}")

        # Make a prediction using the saved image
        pred = prediction(file_path)

        # Fetch the disease info based on the prediction
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]

        # Fetch the supplement info based on the prediction
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]

        # Print the URL to check if it's being passed correctly
        print(f"Supplement Image URL: {supplement_image_url}")

        # Pass the data to the template
        return render_template('submit.html', 
                               title=title, 
                               desc=description, 
                               prevent=prevent, 
                               image_url=image_url, 
                               sname=supplement_name, 
                               simage=supplement_image_url)

# Route for displaying market (supplement) information
@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', 
                           supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']),
                           disease=list(disease_info['disease_name']))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
