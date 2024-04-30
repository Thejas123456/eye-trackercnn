import cv2
import numpy as np
import socket
import struct
import os

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

frame_count = 0

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

        # Save frames to the "no_eyes" folder
        if ID == 1:
            frame_count += 1
            cv2.imwrite(f'no_eyes/frame_{frame_count}.jpg', frames)

        # Stop when it reaches 1000 images
        if frame_count >= 1000:
            print("Reached 1000 images. Stopping.")
            break

    except Exception as e:
        print(f"Error receiving or displaying frame: {e}")

# Close the socket
server.close()
