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
import pygame
import sys
import cv2

def detect_boxes(frame, xcenter, ycenter):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Threshold the HSV image to get only green colors
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Define range of red color in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([160, 100, 100])
    upper_red = np.array([179, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red, upper_red)

    # Combine the masks for red color
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Bitwise-AND mask and original image
    res_green = cv2.bitwise_and(frame, frame, mask=mask_green)
    res_red = cv2.bitwise_and(frame, frame, mask=mask_red)

    # Find contours for green and red boxes
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw green and red boxes
    max_area_green = 0
    max_contour_green = None
    for contour in contours_green:
        area = cv2.contourArea(contour)
        if area > max_area_green:
            max_area_green = area
            max_contour_green = contour

    max_area_red = 0
    max_contour_red = None
    for contour in contours_red:
        area = cv2.contourArea(contour)
        if area > max_area_red:
            max_area_red = area
            max_contour_red = contour

    # Draw the largest green and red boxes
    if max_contour_green is not None:
        x, y, w, h = cv2.boundingRect(max_contour_green)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if max_contour_red is not None:
        x, y, w, h = cv2.boundingRect(max_contour_red)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return frame, (x, y, w, h)

def jump():
    global is_jumping, jump_vel, player_y, jump_timer
    if not is_jumping and jump_timer >= 3:
        is_jumping = True
        jump_vel = -10
        jump_timer = 0

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
model_path = "models/CNN_2_normalized.pth"  # Provide the path to your saved model
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

# Constants
WIDTH = 800
HEIGHT = 600
BLACK = (0, 0, 0)
RED = (255, 0, 0)
FPS = 60

# Player properties
player_width = 50
player_height = 50
player_x = 50
player_y = HEIGHT - player_height
player_vel = 5
jump_vel = -10
gravity = 0.5
is_jumping = False
jump_timer = 0

# Initialize pygame
pygame.init()

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Python Game")

clock = pygame.time.Clock()

# Main game loop
running = True
while running:
    try:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

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

            with torch.no_grad():
                output = model(input_image)
            output_np = output.cpu().numpy()
            # Print the output
            print("Model Output:")
            xcenter = output_np[0][0]
            ycenter = output_np[0][1]
            cv2.imshow('frames', frames)
        elif ID == 2:
            frames2 = imgdecode

            cv2.circle(frames2, (int(xcenter), int(ycenter)), radius=30, color=(0, 255, 0), thickness=-1)
            frames2_with_boxes, (box_x, box_y, box_w, box_h) = detect_boxes(frames2, xcenter, ycenter)
            cv2.imshow('frames2', frames2_with_boxes)
            # Check if gaze is inside red box
            if box_x <= xcenter <= box_x + box_w and box_y <= ycenter <= box_y + box_h:
                jump_timer += 1
                if jump_timer >= 10:  # 180 frames at 60 FPS = 3 seconds
                    jump()
                    print("jump")
                    jump_timer = 0
        # Update player position
        if is_jumping:
            jump_timer = 0
            player_y += jump_vel
            jump_vel += gravity
            if player_y >= HEIGHT - player_height:
                is_jumping = False
                player_y = HEIGHT - player_height
                jump_vel = -10
        else:
            player_x += player_vel
            if player_x >= WIDTH:
                player_x = -player_width

        # Clear the screen
        screen.fill(BLACK)

        # Draw the player
        pygame.draw.rect(screen, RED, (player_x, player_y, player_width, player_height))

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(FPS)

    except Exception as e:
        print(f"Error receiving or displaying frame: {e}")

# Close the socket and window
server.close()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()