import numpy as np
import cv2
import os, sys

import pandas as pd

from pedestrian_orientation import pose_estimator

import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


folder = "images/"

for filename in os.listdir(folder):
    # if it is folder continue
    if os.path.isdir(os.path.join(folder, filename)):
        continue
    img = cv2.imread(os.path.join(folder, filename))
    if img is not None:
        landmarks = pose_estimator.get_pose_landmarks(img)

        if landmarks is None:
            print("No landmarks found in image: ", filename)
            continue

        # add landmarks to dataframe
        df = pd.DataFrame(columns=["x", "y", "z", "visibility", "name"])
        for i, landmark in enumerate(landmarks.landmark):
            df.loc[i] = [
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility,
                mp_pose.PoseLandmark(i).name,
            ]

        # save landmarks to csv
        df.to_csv(os.path.join(folder, filename + ".csv"), index=False)
