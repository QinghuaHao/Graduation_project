import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database') 

class DataCollection:
    def __init__(self, width=640, height=480):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.width = width
        self.height = height
        self.previous_hand_position = None

    def process_video(self, cap):
        landmark_trail = []
        frame_count = 0  # 添加帧计数器
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read frame from video stream")
                break
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_landmarks_data = []
            if results.multi_hand_landmarks:
                if self.previous_hand_position is None:
                    self.previous_hand_position = results.multi_hand_landmarks[
                        0].landmark[self.mp_hands.HandLandmark.WRIST]
                    initial_hand_wrist_coords = (int(
                        self.previous_hand_position.x * frame.shape[1]), int(self.previous_hand_position.y * frame.shape[0]))
                else:
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
                                'frame': frame_count,  # 添加帧编号
                                'hand': 0,
                                'landmark': landmark_id,
                                'x': landmark.x,
                                'y': landmark.y,
                            }
                            frame_landmarks_data.append(landmark_data)
                            if landmark_id == 8:
                                x, y = int(
                                    landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                                if len(landmark_trail) > 0:
                                    cv2.line(
                                        frame, landmark_trail[-1], (x, y), (0, 255, 0), 2)
                                landmark_trail.append((x, y))
                                for pt in landmark_trail:
                                    cv2.circle(frame, pt, 3, (0, 255, 0), -1)
            for i in range(1, len(landmark_trail)):
                cv2.line(frame, landmark_trail[i-1],
                         landmark_trail[i], (0, 255, 0), 2)
            cv2.imshow('Camera Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            df = pd.DataFrame(frame_landmarks_data)
            gesture = self.detect_gesture(df)
            yield df, gesture

            frame_count += 1  # 增加帧计数器

        cap.release()
        cv2.destroyAllWindows()

    def store_file(self, name, df):
        file = f"../data/data_csv/{name}.csv"
        df.to_csv(file, index=False)

    def detect_gesture(self, df):
        if df.empty:
            return 'None'

        landmarks = {row['landmark']: (row['x'], row['y'])
                     for _, row in df.iterrows()}
        if self.is_ok_gesture(landmarks):
            return 'OK'
        elif self.is_victory_gesture(landmarks):
            return 'Victory'
        elif self.is_thumbs_up_gesture(landmarks):
            return 'Thumbs Up'
        elif self.is_thumbs_down_gesture(landmarks):
            return 'Thumbs Down'
        elif self.is_fist_gesture(landmarks):
            return 'Fist'
        elif self.is_five_gesture(landmarks):
            return 'Five'
        else:
            return 'None'

    def is_ok_gesture(self, landmarks):
        if 4 in landmarks and 8 in landmarks:
            thumb_tip = np.array(landmarks[4])
            index_tip = np.array(landmarks[8])
            distance = np.linalg.norm(thumb_tip - index_tip)
            if distance < 0.05:  # Threshold for OK gesture
                return True
        return False

    def is_victory_gesture(self, landmarks):
        if 8 in landmarks and 12 in landmarks:
            index_tip = np.array(landmarks[8])
            middle_tip = np.array(landmarks[12])
            distance = np.linalg.norm(index_tip - middle_tip)
            if distance > 0.1:  # Threshold for Victory gesture
                return True
        return False

    def is_thumbs_up_gesture(self, landmarks):
        if 4 in landmarks and 3 in landmarks and 2 in landmarks:
            thumb_tip = np.array(landmarks[4])
            thumb_ip = np.array(landmarks[3])
            thumb_mcp = np.array(landmarks[2])
            # Check if thumb is pointing upwards
            if thumb_tip[1] < thumb_ip[1] < thumb_mcp[1]:
                return True
        return False

    def is_thumbs_down_gesture(self, landmarks):
        if 4 in landmarks and 3 in landmarks and 2 in landmarks:
            thumb_tip = np.array(landmarks[4])
            thumb_ip = np.array(landmarks[3])
            thumb_mcp = np.array(landmarks[2])
            # Check if thumb is pointing downwards
            if thumb_tip[1] > thumb_ip[1] > thumb_mcp[1]:
                return True
        return False

    def is_fist_gesture(self, landmarks):
        for i in range(5, 21):
            if i in landmarks:
                x, y = landmarks[i]
                if y < landmarks[0][1]:
                    return False
        return True

    def is_five_gesture(self, landmarks):
        if all(i in landmarks for i in range(5, 21)):
            return True
        return False

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
    else:
        data_collector = DataCollection()
        for frame_df, gesture in data_collector.process_video(cap):
            print("Frame landmarks:\n", frame_df)
            print("Detected gestures:", gesture)
            # data_collector.store_file("hand_landmarks", frame_df)
    cap.release()
