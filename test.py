import cv2

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Coordinates (x, y):", x, y)

# Read the image
image = cv2.imread("frames2_folder/0.jpg")

# Display the image
cv2.imshow("Image", image)

# Set mouse callback function
cv2.setMouseCallback("Image", click_event)

# Wait for any key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
