from flask import Flask, render_template, request
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# Initialize Flask app
app = Flask(__name__)

# Define the SimpleCNN model architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32*25*25, 128)
        self.fc2 = nn.Linear(128, 4)  # Update to 3 output classes (chaitanya, thalesh, ansh, avi)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained PyTorch model
model = SimpleCNN()
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Define a function to preprocess the input image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL Image
        transforms.Grayscale(),   # Convert to grayscale
        transforms.Resize((100, 100)),  # Resize image to match training dimensions
        transforms.ToTensor(),           # Convert image to PyTorch tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values
    ])
    image = transform(image)
    return image

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define route for uploading image
@app.route('/', methods=['GET', 'POST'])
# Define route for uploading image
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            
            # Detect faces in the image
            faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # List of individuals whose attendance needs to be marked
            attendance_list = ['ansh', 'chaitanya', 'thalesh', 'avi'] 
            
            # Dictionary to store attendance status
            attendance_status = {name: 'absent' for name in attendance_list}
            
            # Define mapping from predicted class index to name
            class_to_name = {0: 'ansh', 1: 'avi', 2: 'chaitanya', 3: 'thalesh'}

            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract the face region from the image
                face_roi = img[y:y+h, x:x+w]
                
                # Preprocess the face image
                face_tensor = preprocess_image(face_roi)
                
                # Make prediction using the model
                with torch.no_grad():
                    output = model(face_tensor.unsqueeze(0))  # Add batch dimension
                
                # Get predicted label
                _, predicted = torch.max(output, 1)
                
                # Determine name based on predicted class index
                predicted_name = class_to_name[predicted.item()]
                
                # Mark individual as present if recognized
                if predicted_name in attendance_status:
                    attendance_status[predicted_name] = 'Present'
            
            return render_template('result.html', attendance_status=attendance_status)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
