import cv2
import numpy as np
import imutils

def detectshape(c):
    # initialize the shape name and approximate the contour
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
        shape = "triangle"

    # if the shape has 4 vertices, it is either a square or a rectangle
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)

        # a square will have an aspect ratio that is approximately equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

    # if the shape is a pentagon, it will have 5 vertices
    elif len(approx) == 5:
        shape = "pentagon"

    # otherwise, we assume the shape is a circle
    else:
        shape = "circle"

    # return the name of the shape
    return shape

# Load the image
image = cv2.imread("shapeUji8.png")
resized = imutils.resize(image, width=300)

# Calculate the resize ratio
orig_height, orig_width = image.shape[:2]
resized_height = resized.shape[0]
ratio = orig_height / resized_height

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# Find contours and hierarchy in the image and initialize the shape detector
cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on detected shapes (square and triangle)
shapes = []
shapesSgtg = []

for cIdx, c in enumerate(cnts):
    shape = detectshape(c)
    if shape == "square":
        shapes.append(cIdx)
    elif shape == "triangle":
        shapesSgtg.append(cIdx)

# Draw contours and center of the shape on the image
for i, c in enumerate(cnts):
    # Calculate the center of the contour
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
    else:
        cX, cY = 0, 0

    # Multiply the contour (x, y)-coordinates by the resize ratio
    c = c.astype(int) * ratio

    # Check if the contour has at least one point
    if c.size > 0:
        # Draw the contour and center of the shape on the image
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Show the image
cv2.imshow("shape_and_colors.png", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

