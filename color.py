import cv2
import numpy as np

# Capturing Video through webcam.
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    # Converting frame(img) from BGR (Blue-Green-Red) to HSV (Hue-Saturation-Value)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining the range of colors (Blue, Red, and Green)
    blue_lower = np.array([90, 100, 100], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    red_lower = np.array([0, 100, 100], np.uint8)
    red_upper = np.array([10, 255, 255], np.uint8)
    green_lower = np.array([50, 100, 100], np.uint8)
    green_upper = np.array([70, 255, 255], np.uint8)

    # Creating masks for the colors
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Applying morphological transformations (Dilation)
    kernal = np.ones((5, 5), "uint8")
    blue_res = cv2.bitwise_and(img, img, mask=blue_mask)
    red_res = cv2.bitwise_and(img, img, mask=red_mask)
    green_res = cv2.bitwise_and(img, img, mask=green_mask)

    # Tracking color (Blue)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in blue_contours:
        area = cv2.contourArea(contour)
        if area > 900:
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(img, "Blue", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Tracking color (Red)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in red_contours:
        area = cv2.contourArea(contour)
        if area > 900:
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(img, "Red", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Tracking color (Green)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in green_contours:
        area = cv2.contourArea(contour)
        if area > 900:
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, "Green", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show color detection result
    cv2.imshow("Color Tracking", img)
    cv2.imshow("Blue", blue_res)
    cv2.imshow("Red", red_res)
    cv2.imshow("Green", green_res)

    # Exit if 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
