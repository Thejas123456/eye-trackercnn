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


HOST = '192.168.197.45'
PORT = 9999
buffSize = 65535
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model_path = "models/CNN_2_normalized_200epochs.pth"  # Provide the path to your saved model
model.load_state_dict(torch.load(model_path))
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.389 , 0.389, 0.389], std=[0.207, 0.207, 0.207]),  # Normalize
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

            # Extract feature maps from convolutional layers
            with torch.no_grad():
                conv1_features = F.relu(model.conv1(input_image))
                conv2_features = F.relu(model.conv2(conv1_features))
                conv3_features = F.relu(model.conv3(conv2_features))
                conv4_features = F.relu(model.conv4(conv3_features))
                output = model(input_image)

            # Upsample feature maps to match the input image size
            upsample = nn.Upsample(size=(frames.shape[0], frames.shape[1]), mode='bilinear', align_corners=True)
            upsampled_features = upsample(conv4_features)

            # Create a mask from the upsampled features
            mask = upsampled_features.mean(dim=1, keepdim=True)  # Take mean across channels
            mask = mask.squeeze().cpu().numpy()  # Convert to numpy array

            # Normalize the mask
            mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
            mask = (mask * 255).astype(np.uint8)

            # Overlay the mask onto the frames
            mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            overlaid_frames = cv2.addWeighted(frames, 0.5, mask, 0.5, 0)
            output_np = output.cpu().numpy()
            xcenter = output_np[0][0]
            ycenter = output_np[0][1]
            # Display the frames with overlaid mask
            cv2.imshow('frames', overlaid_frames)
        elif ID == 2:
            frames2 = imgdecode
            
            cv2.circle(frames2, (int(xcenter), int(ycenter)), radius=30, color=(0, 255, 0), thickness=-1)
            cv2.imshow('frames2', imgdecode)
        # Press "ESC" to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break
    except Exception as e:
        print(f"Error receiving or displaying frame: {e}")

# Close the socket and window
server.close()
cv2.destroyAllWindows()
