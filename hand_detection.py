import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Callback function for trackbars (does nothing)
def nothing(x):
    pass

# Create a window for color adjustment trackbars
cv2.namedWindow("Color Adjustments", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Color Adjustments", (300, 300))

# Trackbars for lower HSV bounds
cv2.createTrackbar("Lower_H", "Color Adjustments", 0, 180, nothing)
cv2.createTrackbar("Lower_S", "Color Adjustments", 48, 255, nothing)
cv2.createTrackbar("Lower_V", "Color Adjustments", 80, 255, nothing)

# Trackbars for upper HSV bounds
cv2.createTrackbar("Upper_H", "Color Adjustments", 20, 180, nothing)
cv2.createTrackbar("Upper_S", "Color Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper_V", "Color Adjustments", 255, 255, nothing)

# Thresholding trackbar
cv2.createTrackbar("Thresh", "Color Adjustments", 0, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for consistency
    frame = cv2.resize(frame, (400, 400))

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get HSV values from trackbars
    l_h = cv2.getTrackbarPos("Lower_H", "Color Adjustments")
    l_s = cv2.getTrackbarPos("Lower_S", "Color Adjustments")
    l_v = cv2.getTrackbarPos("Lower_V", "Color Adjustments")
    u_h = cv2.getTrackbarPos("Upper_H", "Color Adjustments")
    u_s = cv2.getTrackbarPos("Upper_S", "Color Adjustments")
    u_v = cv2.getTrackbarPos("Upper_V", "Color Adjustments")

    # Set HSV thresholds
    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    # Create the mask and apply Gaussian blur to reduce noise
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Filter original frame using the mask
    filtered = cv2.bitwise_and(frame, frame, mask=mask)

    # Apply binary threshold
    thresh_val = cv2.getTrackbarPos("Thresh", "Color Adjustments")
    _, thresh = cv2.threshold(mask, thresh_val, 255, cv2.THRESH_BINARY)

    # Dilate thresholded image to fill in gaps
    dilated = cv2.dilate(thresh, (3, 3), iterations=7)

    # Detect contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours and convex hulls
    for contour in contours:
        epsilon = 0.0001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        hull = cv2.convexHull(approx)

        cv2.drawContours(frame, [contour], -1, (50, 50, 150), 2)
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)

    # Display all relevant windows
    cv2.imshow("Mask", mask)
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Filtered", filtered)
    cv2.imshow("Final Output", frame)

    # Exit loop when 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

