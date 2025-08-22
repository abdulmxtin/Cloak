import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
time.sleep(2)

bg = None
for i in range(60):
    ret, frame = cap.read()
    if not ret:
        continue
    frame = np.flip(frame, axis=1)
    if bg is None:
        bg = frame.astype('float')
    else:
        cv2.accumulateWeighted(frame, bg, 0.5)
bg = cv2.convertScaleAbs(bg)

print("Background captured. Now put your cloak in front of camera.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = np.flip(frame, axis=1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
    mask_inv = cv2.bitwise_not(mask)

    part1 = cv2.bitwise_and(bg, bg, mask=mask)
    part2 = cv2.bitwise_and(frame, frame, mask=mask_inv)

    final = cv2.addWeighted(part1, 1, part2, 1, 0)

    cv2.imshow('Invisibility Cloak - Press q to quit', final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
