import cv2              # Import the OpenCV library
import numpy as np      # Import the NumPy library
import torch            # Import the PyTorch library

# Load the YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Start the camera
cap = cv2.VideoCapture(0)

# Read the first frame and flip it horizontally
_, prev = cap.read()
prev = cv2.flip(prev, 1)

# Read the new frame and flip it horizontally
_, new = cap.read()
new = cv2.flip(new, 1)

# Define the reference point (a point in the top-right corner of the screen)
ref_point = (prev.shape[1] - 10, 10)

# Define screen and camera resolutions
screen_resolution = (1920, 1080)
camera_resolution = (1280, 720)

# Calculate the conversion rate from pixels to centimeters
pixel_to_cm = 0.026458333333333 * (camera_resolution[0] / screen_resolution[0])

# Function to calculate distance from the object's center using Pythagorean theorem
def calculate_distance(object_center):
    distance_px = ((object_center[0] - ref_point[0]) ** 2 + (object_center[1] - ref_point[1]) ** 2) ** 0.5
    distance_cm = distance_px * pixel_to_cm
    return distance_cm

# Infinite loop
while True:
    diff = cv2.absdiff(prev, new)                                                 # Calculate the difference between two frames
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)                                 # Convert the difference to grayscale
    diff = cv2.blur(diff, (5, 5))                                                 # Apply blurring
    _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)                   # Apply binary thresholding
    thresh = cv2.dilate(thresh, None, 3)                                          # Apply dilation
    thresh = cv2.erode(thresh, np.ones((4, 4)), 1)                                # Apply erosion
    contor, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    
    # Detect objects in the previous frame using YOLOv5
    results = model(prev)
    boxes = results.pred[0][:, :4]  # Get the bounding boxes of the detected objects

    # For each detected object
    for box in boxes:
        x_min, y_min, x_max, y_max = box.cpu().numpy().astype(int)  # Get the bounding box coordinates
        
        # Keep the coordinates within the valid range
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, prev.shape[1])
        y_max = min(y_max, prev.shape[0])

        # Draw the bounding box
        cv2.rectangle(prev, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Calculate the center point of the object
        object_center_x = (x_min + x_max) // 2
        object_center_y = (y_min + y_max) // 2

        # Draw the reference point and the object center
        cv2.circle(prev, ref_point, 5, (0, 0, 255), -1)
        cv2.circle(prev, (object_center_x, object_center_y), 5, (0, 0, 255), -1)

        # Calculate the distance and display it on the screen
        distance_cm = calculate_distance((object_center_x, object_center_y))
        cv2.putText(prev, "{:.2f} cm".format(distance_cm), (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the result
    cv2.imshow("Object Distance Calculation from Camera", prev)

    # Set the new frame as the previous frame and read the next frame
    prev = new
    _, new = cap.read()
    new = cv2.flip(new, 1)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()