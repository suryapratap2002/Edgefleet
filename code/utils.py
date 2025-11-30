import os
import cv2
import numpy as np

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def ball_hsv_mask(frame, lower=(5,90,80), upper=(30,255,255)):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(lower); upper = np.array(upper)
    mask = cv2.inRange(hsv, lower, upper)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def largest_contour_centroid(mask, min_area=10):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None
    M = cv2.moments(c)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00']); cy = int(M['m01'] / M['m00'])
    return cx, cy
