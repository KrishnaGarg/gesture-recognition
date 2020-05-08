import cv2
from sklearn.metrics import pairwise
import numpy as np
import imutils

background = None
threshold=25

def setup(camera, top, right, bottom, left):
    frame = camera.read()[1]
    frame = imutils.resize(frame, width=700)
    frame = cv2.flip(frame, 1)
    radiusOfInterest = frame[top:bottom, right:left]
    frameCopy = frame.copy()
    return radiusOfInterest, frameCopy

def computeAverage(alpha, image):
    global background
    subCopy = image.copy()
    if background is not None:
        cv2.accumulateWeighted(image, background, alpha)
    else:
        print("No background detected")
        background = subCopy.astype("float")

def segment(image):
    global background
    global threshold
    residue = cv2.absdiff(background.astype("uint8"), image)
    filteredOut = cv2.threshold(residue, threshold, 255, cv2.THRESH_BINARY)[1]
    numOfContours = cv2.findContours(filteredOut.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    if len(numOfContours) != 0:
        segmented = max(numOfContours, key=cv2.contourArea)
        return (filteredOut, segmented)
    else:
        return

def count(filteredOut, segmented):
    polygon = cv2.convexHull(segmented)
    left   = tuple(polygon[polygon[:, :, 0].argmin()][0])
    right  = tuple(polygon[polygon[:, :, 0].argmax()][0])
    top    = tuple(polygon[polygon[:, :, 1].argmin()][0])
    bottom = tuple(polygon[polygon[:, :, 1].argmax()][0])

    center_X = int((left[0] + right[0]) / 2)
    center_Y = int((top[1] + bottom[1]) / 2)

    distance = pairwise.euclidean_distances([(center_X, center_Y)], Y=[left, right, top, bottom])[0]
    maximum_distance = distance[distance.argmax()]

    radius = int(0.8 * maximum_distance)
    circular_roi = np.zeros(filteredOut.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (center_X, center_Y), radius, 255, 1)
    circular_roi = cv2.bitwise_and(filteredOut, filteredOut, mask=circular_roi)
    numOfContours = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

    count = 0
    circumference = (2 * np.pi * radius)

    for c in numOfContours:
        y = cv2.boundingRect(c)[1]
        h = cv2.boundingRect(c)[3]
        if ((center_Y + (center_Y * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count

if __name__ == "__main__":
    alpha = 0.5
    top = 10
    right = 350
    bottom = 225
    left = 590
    camera = cv2.VideoCapture(0)
    frameCount = 0
    calibrated = False

    while(True):
        radiusOfInterest, frameCopy = setup(camera, top, right, bottom, left)
        gray = cv2.cvtColor(radiusOfInterest, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if frameCount >= 30:
            hand = segment(gray)

            if hand is not None:
                (filteredOut, segmented) = hand
                cv2.drawContours(frameCopy, [segmented + (right, top)], -1, (0, 0, 255))
                countOfFingers = count(filteredOut, segmented)
                cv2.putText(frameCopy, str(countOfFingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow("After background subtraction", filteredOut)
        else:
            computeAverage(alpha, gray)
            if frameCount == 1:
                print("Calibration in progress")
            elif frameCount == 29:
                print("Calibration successful")
        
        cv2.rectangle(frameCopy, (left, top), (right, bottom), (255,0,0), 2)
        frameCount += 1
        cv2.imshow("Video stream", frameCopy)
        keyboardInput = cv2.waitKey(1) & 0xFF
        if keyboardInput == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()