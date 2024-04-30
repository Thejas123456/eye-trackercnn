
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

# Create folders to save images and coordinates
os.makedirs("frames_folder", exist_ok=True)
os.makedirs("frames2_folder", exist_ok=True)
os.makedirs("coordinates", exist_ok=True)

coordinate_count = 0

# Function to handle mouse click event
def save_coordinates(event, x, y, flags, param):
    global coordinate_count
    if event == cv2.EVENT_LBUTTONDOWN:
        # Save the coordinates in a text file
        with open(f'coordinates/coordinates_{coordinate_count}.txt', 'w') as f:
            f.write(f"{x} {y}\n")

        # Save the images from frames and frames3
        cv2.imwrite(f'frames_folder/{coordinate_count}.jpg', frames)
        cv2.imwrite(f'frames2_folder/{coordinate_count}.jpg', frames2)
        coordinate_count += 1

cv2.namedWindow('frames2')
cv2.setMouseCallback('frames2', save_coordinates)

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
        data_chunks = bytearray()
        received_length = 0
        while received_length < length:
            data, address = server.recvfrom(min(buffSize, length - received_length))
            data_chunks.extend(data)
            received_length += len(data)

        # Decode image
        imgdecode = cv2.imdecode(np.asarray(data_chunks), 1)

        # Display image based on ID
        if ID == 1:
            frames = imgdecode
            frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
            cv2.imshow('frames', frames)
        elif ID == 2:
            frames2 = imgdecode
            cv2.imshow('frames2', imgdecode)

        # Press "ESC" to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break
    except Exception as e:
        print(f"Error receiving or displaying frame: {e}")

# Close the socket and window
server.close()
cv2.destroyAllWindows()
