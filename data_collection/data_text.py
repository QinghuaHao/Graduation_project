import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import datetime
import warnings
import absl.logging
import random

absl.logging.set_verbosity(absl.logging.INFO)


class DataCollection:
    def __init__(self, width=640, height=480, sequence_length=60):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.width = width
        self.height = height
        self.recording = False
        self.previous_hand_position = None
        self.initial_hand_wrist_coords = None
        self.sequence_length = sequence_length
        self.recorded_data = []
        self.landmark_trail = []  # 用于记录landmark 8的轨迹
        self.standard_gestures_dir = '../data/standard_gestures/circle'#修改数据位置
        if not os.path.exists(self.standard_gestures_dir):
            os.makedirs(self.standard_gestures_dir)

    def start_recording(self):
        self.recording = True
        self.recorded_data = []
        self.landmark_trail = []

    def stop_recording(self):
        self.recording = False
        if self.recorded_data:
            df = pd.DataFrame(self.recorded_data)
            normalized_df = self.normalize_sequence_length(df)
            normalized_df['time'] = range(len(normalized_df))  # 增加time列
            self.save_standard_gesture(normalized_df)
        self.landmark_trail = []

    def normalize_sequence_length(self, df):
        while len(df) < self.sequence_length:
            insert_idx = random.randint(1, len(df) - 1)
            new_row = (df.iloc[insert_idx - 1][['x', 'y']] + df.iloc[insert_idx][['x', 'y']]) / 2
            new_row = pd.Series(new_row, index=df.columns)
            df = pd.concat([df.iloc[:insert_idx], pd.DataFrame([new_row]), df.iloc[insert_idx:]], ignore_index=True)
        return df.reset_index(drop=True)

    def save_standard_gesture(self, df):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(
            self.standard_gestures_dir,
            f"{timestamp}.csv")
        df.to_csv(file_path, index=False)
        print(f"Gesture saved to {file_path}")

    def process_frame(self, frame, frame_count):
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_landmarks_data = []

        if results.multi_hand_landmarks:
            if self.previous_hand_position is None:
                self.previous_hand_position = results.multi_hand_landmarks[
                    0].landmark[self.mp_hands.HandLandmark.WRIST]
                self.initial_hand_wrist_coords = (int(
                    self.previous_hand_position.x * frame.shape[1]), int(self.previous_hand_position.y * frame.shape[0]))
            else:
                min_distance = float('inf')
                closest_hand_landmarks = None
                for hand_landmarks in results.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                    wrist_coords = (
                        int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]))
                    distance = np.linalg.norm(
                        np.array(
                            self.initial_hand_wrist_coords) -
                        np.array(wrist_coords))
                    if distance < min_distance:
                        min_distance = distance
                        closest_hand_landmarks = hand_landmarks
                if closest_hand_landmarks:
                    self.previous_hand_position = closest_hand_landmarks.landmark[
                        self.mp_hands.HandLandmark.WRIST]
                    self.mp_drawing.draw_landmarks(
                        frame, closest_hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    for landmark_id, landmark in enumerate(
                            closest_hand_landmarks.landmark):
                        landmark_data = {
                            'frame': frame_count,
                            'hand': 0,  # or other value depending on your requirement
                            'landmark': landmark_id,
                            'x': landmark.x,
                            'y': landmark.y,
                        }
                        frame_landmarks_data.append(landmark_data)
                        if self.recording and landmark_id == 8:
                            self.recorded_data.append(landmark_data)
                            x, y = int(landmark.x *
                                       frame.shape[1]), int(landmark.y *
                                                            frame.shape[0])
                            self.landmark_trail.append((x, y))

        # 绘制轨迹
        if self.recording and len(self.landmark_trail) > 1:
            for i in range(1, len(self.landmark_trail)):
                cv2.line(frame,
                         self.landmark_trail[i - 1],
                         self.landmark_trail[i],
                         (0,
                          255,
                          0),
                         2)

        return frame, frame_landmarks_data

    def process_video(self, cap):
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read frame from video stream")
                break

            frame, frame_landmarks_data = self.process_frame(
                frame, frame_count)
            cv2.imshow('Camera Feed', frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('r'):
                self.start_recording()
                print("Started recording...")
            elif key & 0xFF == ord('s'):
                self.stop_recording()

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Cannot open camera")
    else:
        data_collector = DataCollection()
        data_collector.process_video(cap)
    cap.release()
