import cv2
from h11 import Data
import mediapipe as mp
import time
import numpy as np
import pandas as pd
# determine hand_id


class DataCollection():

    def __init__(self, camera_num=0, width=640, height=480, fps=30, duration=10, num_points=40):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.landmarks_data = []
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration
        self.previous_hand_position = None
        self.num_points = num_points
        self.camera_num = camera_num

    def open_camera(self):
        cap = cv2.VideoCapture(self.camera_num)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not cap.isOpened():
            print("can not open camera")
            return

        start_time = time.time()
        frame_count = 0
        landmark_trail = []
        points_collected = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("camera can not open")
                break
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                if self.previous_hand_position is None:
                    # Record the initial hand position (wrist coordinates)
                    self.previous_hand_position = results.multi_hand_landmarks[
                        0].landmark[self.mp_hands.HandLandmark.WRIST]
                    initial_hand_wrist_coords = (int(
                        self.previous_hand_position.x * frame.shape[1]), int(self.previous_hand_position.y * frame.shape[0]))
                else:
                    # Find the hand closest to the previous hand position
                    min_distance = float('inf')
                    closest_hand_landmarks = None
                    for hand_landmarks in results.multi_hand_landmarks:
                        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                        wrist_coords = (
                            int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]))
                        distance = np.linalg.norm(
                            np.array(initial_hand_wrist_coords) - np.array(wrist_coords))
                        if distance < min_distance:
                            min_distance = distance
                            closest_hand_landmarks = hand_landmarks
                    if closest_hand_landmarks:
                        self.previous_hand_position = closest_hand_landmarks.landmark[
                            self.mp_hands.HandLandmark.WRIST]
                        self.mp_drawing.draw_landmarks(
                            frame, closest_hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        for landmark_id, landmark in enumerate(closest_hand_landmarks.landmark):
                            landmark_data = {
                                'frame': frame_count,
                                'hand': 0,  # Always 0 since we're only tracking one hand
                                'landmark': landmark_id,
                                'x': landmark.x,
                                'y': landmark.y,
                            }
                            self.landmarks_data.append(landmark_data)
                            if landmark_id == 8:
                                x, y = int(
                                    landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                                if len(landmark_trail) > 0:
                                    cv2.line(
                                        frame, landmark_trail[-1], (x, y), (0, 255, 0), 2)
                                landmark_trail.append((x, y))
                                for pt in landmark_trail:
                                    cv2.circle(frame, pt, 3, (0, 255, 0), -1)
                                points_collected += 1
                                df = pd.DataFrame(self.landmarks_data)
                                if points_collected >= self.num_points:
                                    break
                        if points_collected >= self.num_points:
                            break
                    if points_collected >= self.num_points:
                        break
            for i in range(1, len(landmark_trail)):
                cv2.line(frame, landmark_trail[i-1],
                         landmark_trail[i], (0, 255, 0), 2)
            cv2.imshow('Camera Feed', frame)
            # print(self.landmarks_data)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_count += 1
            if time.time() - start_time > self.duration:
                break
        cap.release()
        cv2.destroyWindow('Camera Feed')
        return df

    def store_file(self, name):
        file = f"../data/data_csv/{name}.csv"
        self.open_camera().to_csv(file)


if __name__ == '__main__':
    dc = DataCollection(width=640, height=480, fps=60,
                        duration=10, num_points=80)
    dc.store_file(name=1)
