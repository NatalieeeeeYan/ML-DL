# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:27:05 2024

@author: PC
"""

import cv2



cap = cv2.VideoCapture(0)
while (1):
    ret, frame = cap.read()

    cv2.imshow('frame',frame)
  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()