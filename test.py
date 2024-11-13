import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN model with 4 output classes
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32*25*25, 128)
        self.fc2 = nn.Linear(128, 4)  # Updated to 4 output classes

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
        transforms.Resize((100, 100)),  # Resize image to match training dimensions
        transforms.ToTensor(),           # Convert image to PyTorch tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values
    ])
    image = Image.fromarray(image)
    image = transform(image)
    return image

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video from the front camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face_roi = gray[y:y+h, x:x+w]
        
        # Preprocess the face image
        face_tensor = preprocess_image(face_roi)
        
        # Make prediction using the model
        with torch.no_grad():
            output = model(face_tensor.unsqueeze(0))  # Add batch dimension
            
        # Get predicted label and confidence
        _, predicted = torch.max(output, 1)
        confidence = torch.softmax(output, 1)[0, predicted.item()].item()
        
        # Determine label based on predicted index
        if predicted.item() == 0:
            label = 'ansh'
        elif predicted.item() == 1:
            label = 'avi'
        elif predicted.item() == 2:
            label = 'chaitanya'
        elif predicted.item() == 3:
            label = 'thalesh'
        else:
            label = 'unknown'

        # Display label and confidence on the frame
        cv2.putText(frame, f'{label} ({confidence:.2f})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow('Face Recognition', frame)
    
    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
