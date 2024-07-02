import cv2
import mediapipe as mp
import time


class dataCollection():

    def __init__(self, width=640, height=480, fps=30, duration=10):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils
        self.landmarks_data = []
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration

    def open_camera(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not cap.isOpened():
            print("can not open camera")
            return

        start_time = time.time()
        frame_count = 0
        landmark_trail = []


        while True:
            ret, frame = cap.read()
            if not ret:
                print("camera can not open")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                        landmark_data = {
                            'frame': frame_count,
                            'hand': hand_id,
                            'landmark': landmark_id,
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        }
                        self.landmarks_data.append(landmark_data)
                        if landmark_id ==8:
                            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                            if len(landmark_trail) > 0:
                                cv2.line(frame, landmark_trail[-1], (x, y), (0, 255, 0), 2)
                            landmark_trail.append((x, y))
                            for pt in landmark_trail:
                                cv2.circle(frame, pt, 3, (0, 255, 0), -1)
                        yield landmark_data
            for i in range(1,len(landmark_trail)):
                cv2.line(frame, landmark_trail[i-1], landmark_trail[i], (0, 255, 0), 2)
            cv2.imshow('Camera Feed', frame)
            # print(self.landmarks_data)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_count += 1
            if time.time() - start_time > self.duration:
                break
        cap.release()
        cv2.destroyWindow('Camera Feed')


if __name__ == '__main__':
    dc = dataCollection(width=640, height=480, fps=60, duration=4)
    for landmark in dc.open_camera():
        print(landmark)
