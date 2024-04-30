import cv2 as cv
import numpy as np
import os

# Path to the folder containing the image
folder_path = "frames_folder/"

# Filename of the image to process
image_filename = "0.jpg"

# Construct the full path to the image
image_path = os.path.join(folder_path, image_filename)

# Read the image
frame = cv.imread(image_path)

if frame is None:
    print(f"Could not read the image {image_filename}")
    exit()

# Convert the image to grayscale
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and improve Hough Circle detection
blurred = cv.GaussianBlur(gray, (5, 5), 0)
_, threshold_eye = cv.threshold(blurred, 65, 255, cv.THRESH_BINARY_INV)
cv.imshow("thres",threshold_eye)
threshold_eye = cv.erode(threshold_eye, None, iterations=2)
threshold_eye = cv.dilate(threshold_eye, None, iterations=4)
threshold_eye = cv.medianBlur(threshold_eye, 5)
cv.imshow("thres2",threshold_eye)
# Detect circles using Hough Circle Transform
circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=30, minRadius=20, maxRadius=100)

# Check if circles are detected
if circles is not None:
    # Convert coordinates and radius to integers
    circles = np.round(circles[0, :]).astype("int")

    # Loop over all detected circles
    for (x, y, r) in circles:
        # Draw the circle on the original image
        cv.circle(frame, (x, y), r, (0, 255, 0), 4)

    print(f"Eye is present in {image_filename}")
else:
    print(f"No eye is present in {image_filename}")

# Display the image with detected circles
cv.imshow('Detected Circles', frame)
cv.waitKey(0)
cv.destroyAllWindows()
