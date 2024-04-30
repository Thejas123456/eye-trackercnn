import cv2
import numpy as np
import socket
import struct
import os
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# Define transformations to apply to images
transform = transforms.Compose([
    transforms.ToPILImage(),        # Convert numpy array to PIL Image
    transforms.Resize((100, 100)),  # Resize images to a fixed size
    transforms.ToTensor(),           # Convert images to PyTorch tensors
])

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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

model = CNN().to(device)
model.load_state_dict(torch.load("eye_det_model.pth"))
model.eval()

HOST = '192.168.197.45'
PORT = 9999
buffSize = 65535

# Create a UDP socket
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

try:
    # Bind the address and port
    server.bind((HOST, PORT))
    print('Now waiting for frames...')
except socket.error as e:
    print(f"Failed to bind server: {e}")
    exit()

# Create a folder to save images without eyes
os.makedirs("no_eyes", exist_ok=True)

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
            cv2.imshow("frames", frames)

            # Convert the frame to RGB format
            frames_rgb = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)

            # Apply transformations
            img_tensor = transform(frames_rgb).unsqueeze(0).to(device)

            # Pass the frame through the model
            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs.data, 1)

            # Print whether an eye is present or not
            if predicted.item() == 0:
                print("Eye is present")
            else:
                print("No eye is present")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error receiving or displaying frame: {e}")

# Close the socket and OpenCV windows
server.close()
cv2.destroyAllWindows()
