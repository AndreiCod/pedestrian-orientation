import numpy as np
import cv2
import os, sys

import pandas as pd

from pedestrian_orientation import pose_estimator

import mediapipe as mp

from pedestrian_orientation.classifier import classify_by_hip_distance

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


folder = "images/cvpr10_multiview_pedestrians/test"

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

        hips_label = classify_by_hip_distance(landmarks)
        print(hips_label)

        # show image
        cv2.imshow("image", img)

        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
        ) as pose:
            image_height, image_width, _ = img.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            mp_drawing.plot_landmarks(
                results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS
            )

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # add landmarks to dataframe
        # df = pd.DataFrame(columns=["x", "y", "z", "visibility", "name"])
        # for i, landmark in enumerate(landmarks.landmark):
        #     df.loc[i] = [
        #         landmark.x,
        #         landmark.y,
        #         landmark.z,
        #         landmark.visibility,
        #         mp_pose.PoseLandmark(i).name,
        #     ]
