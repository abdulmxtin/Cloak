# Invisibility Cloak using OpenCV

## Overview

This project implements an "Invisibility Cloak" effect using OpenCV and Python. The code captures the background, detects a specific color (in this case, red), and replaces the detected region with the background, making it appear as if the object (cloak) is invisible.

## How It Works

1. **Background Capture:**  
   The program begins by capturing and averaging several frames to get a stable background image without the subject in front of the camera.

2. **Live Video Processing:**  
   Once the background is captured, the program processes each frame in real-time:
   - Flips the frame horizontally for a natural mirror effect.
   - Converts the frame from BGR to HSV color space to facilitate color detection.
   - Detects the color of the cloak (red) using two HSV ranges (to cover all shades of red).
   - Refines the mask using morphological operations to remove noise and fill gaps.
   - Separates the cloak area (where the mask is present) and the rest of the frame.
   - Combines the background for the cloak area and the current frame for the rest, creating the invisibility effect.
   - Displays the final output.

3. **Exit:**  
   The program runs in a loop until the user presses the `q` key to quit.

## Code Flow Explanation

### 1. Import Libraries

```python
import cv2
import numpy as np
import time
```
- **cv2:** OpenCV library for image and video processing.
- **numpy:** For numerical operations and creating arrays.
- **time:** For adding delays.

### 2. Capture Video and Background

```python
cap = cv2.VideoCapture(0)
time.sleep(2)
```
- Opens the default webcam.
- Waits for 2 seconds to allow the camera to adjust.

#### Background Initialization

```python
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
```
- Captures 60 frames to compute the background.
- Frames are flipped for mirror effect.
- Accumulates frames to get a smooth background using weighted averaging.

### 3. Real-Time Cloak Effect

```python
print("Background captured. Now put your cloak in front of camera.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = np.flip(frame, axis=1)
```
- Starts processing live video frames.
- Flips each frame horizontally.

#### Convert to HSV and Create Masks

```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 | mask2
```
- Converts frame to HSV color space.
- Defines two ranges for red color (because red wraps in HSV).
- Creates masks for both ranges and combines them.

#### Refine Mask

```python
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
mask_inv = cv2.bitwise_not(mask)
```
- Removes noise and smooths the mask using morphological operations.
- Creates an inverse mask for the non-cloak area.

#### Segment & Combine

```python
part1 = cv2.bitwise_and(bg, bg, mask=mask)
part2 = cv2.bitwise_and(frame, frame, mask=mask_inv)
final = cv2.addWeighted(part1, 1, part2, 1, 0)
```
- `part1`: Area where cloak is detected, filled with background.
- `part2`: Remaining area, shows the live frame.
- Combines both for the invisibility effect.

#### Display Output

```python
cv2.imshow('Invisibility Cloak - Press q to quit', final)
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```
- Shows the final output in a window.
- Exits the loop when 'q' is pressed.

### 4. Cleanup

```python
cap.release()
cv2.destroyAllWindows()
```
- Releases the webcam and closes all windows.

## How to Run

1. Install required libraries:
   ```
   pip install opencv-python numpy
   ```
2. Save the code in a Python file (e.g., `invisibility_cloak.py`).
3. Run the file:
   ```
   python invisibility_cloak.py
   ```
4. Stand out of the camera view for a few seconds while the background is captured.
5. Put on your red cloak and step in front of the camera to see the invisibility effect.
6. Press `q` to exit.

## Notes

- The cloak should be a solid red color for best results.
- You can adjust the HSV color ranges to detect other colors.
- Good lighting and a plain background improve the effect.

## License

This project is for educational purposes.
