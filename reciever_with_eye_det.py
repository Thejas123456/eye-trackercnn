import cv2
import numpy as np
import socket
import struct
import os
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import easyocr

HOST = '192.168.197.45'
PORT = 9999
buffSize = 65535
reader = easyocr.Reader(['en'])
def detect_red_boxes_with_white(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define red color range in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    # Threshold the image to detect red regions
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    #cv2.imshow("mask",mask_red)
    
    # Find contours for red regions
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter out contours based on area and aspect ratio
    red_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 200 and 0.5 < w / h < 2 :
            red_boxes.append((x, y, w, h))
    # Check for white color inside the detected red boxes
    for box in red_boxes:
        x, y, w, h = box
        # Extract the region of interest (ROI) from the original image
        roi = image[y:y+h, x:x+w]
        
        # Convert the ROI to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        # Apply thresholding to detect white color inside the ROI
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        # Invert the thresholded image
        thresh = cv2.bitwise_not(thresh)
        h, w = thresh.shape[:2]
        crop_start = h // 4
        crop_end = 3* h //4
        crop_start2 = w // 4
        crop_end2 = 3* w //4
        thresh = thresh[crop_start:crop_end, crop_start2:crop_end2]
        thresh = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("thresh", thresh)
        # Use Tesseract OCR to detect text
        text = reader.readtext(thresh)  # Adjust page segmentation mode as needed
        # Print the detected text
        print("Text detected:", text)
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)  # Output layer for coordinates
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 256 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, 2)  # Output layer with 2 classes: eye or no_eye

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model_path = "models/CNN_2_normalized_200epochs.pth"  # Provide the path to your saved model
model.load_state_dict(torch.load(model_path))
model.eval()
model2 = CNN2().to(device)
model2.load_state_dict(torch.load("eye_det_model.pth"))
model2.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.389 , 0.389, 0.389], std=[0.207, 0.207, 0.207]),  # Normalize
])
transform2 = transforms.Compose([
    transforms.ToPILImage(),        # Convert numpy array to PIL Image
    transforms.Resize((100, 100)),  # Resize images to a fixed size
    transforms.ToTensor(),           # Convert images to PyTorch tensors
])


# Create a UDP socket
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

try:
    # Bind the address and port
    server.bind((HOST, PORT))
    print('Now waiting for frames...')
except socket.error as e:
    print(f"Failed to bind server: {e}")
    exit()

# Create folders to save images and coordinates
os.makedirs("frames_folder", exist_ok=True)
os.makedirs("frames2_folder", exist_ok=True)
os.makedirs("coordinates", exist_ok=True)

coordinate_count = 0

# Function to handle mouse click event

xcenter = 0 
ycenter = 0
cv2.namedWindow('frames2')



while True:
    try:
        # Receive ID
        data, address = server.recvfrom(buffSize)
        if len(data) != 1:
            continue
        ID = struct.unpack('B', data)[0]

        # Receive length
        data, address = server.recvfrom(buffSize)
        if len(data) != 4:
            continue
        length = struct.unpack('i', data)[0]

        # Receive data
        data, address = server.recvfrom(buffSize)
        if length != len(data):
            continue

        # Decode image
        data = np.array(bytearray(data))
        imgdecode = cv2.imdecode(data, 1)

        # Display image based on ID
        if ID == 1:
            frames = imgdecode
            frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
           # Preprocess the image for model input
            input_image = transform(Image.fromarray(frames)).unsqueeze(0).to(device)
            frames_rgb = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
            input_image2 = transform2(frames_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_image)
                outputs = model2(input_image2)
                _, predicted = torch.max(outputs.data, 1)
                print(outputs)
            output_np = output.cpu().numpy()
            print(output_np[0])
            # Print the output
            print("Model Output:")
            xcenter = output_np[0][0]
            ycenter = output_np[0][1]
            if predicted.item() == 0:
                print("Eye is present")
            else:
                print("No eye is present")
            #print(output_np)
            cv2.imshow('frames', frames)
        elif ID == 2:
            frames2 = imgdecode
            detect_red_boxes_with_white(frames2)
            if(predicted.item() == 0) :
                cv2.circle(frames2, (int(xcenter), int(ycenter)), radius=30, color=(0, 255, 0), thickness=-1)
            cv2.imshow('frames2', frames2)
        # Press "ESC" to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break
    except Exception as e:
        print(f"Error receiving or displaying frame: {e}")

# Close the socket and window
server.close()
cv2.destroyAllWindows()
