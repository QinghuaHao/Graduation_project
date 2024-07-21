import numpy as np
import pandas as pd
import cv2
import time
import os
import sys
from action_association_analysis.action_association_analysis import MotionCorrelationEnhanced
from data_collection.data_collection import DataCollection

file = os.getcwd()
sys.path.append(file)


def load_standard_gestures():
    gestures = {}
    standard_gestures_dir='data/standard_gestures'
    for file_name in os.listdir(standard_gestures_dir):
        if file_name in ['square.csv', 'cirle.csv']:
            gesture_name = file_name[:-4]  # 移除 .csv 后缀
            df = pd.read_csv(os.path.join(standard_gestures_dir, file_name))
            gestures[gesture_name] = df[['x', 'y']]
    return gestures

def draw_trajectory(frame, recorded_data, frame_width, frame_height):
    for i in range(1, len(recorded_data)):
        if recorded_data[i - 1] is None or recorded_data[i] is None:
            continue
        pt1 = (int(recorded_data[i - 1]['x'] * frame_width),
               int(recorded_data[i - 1]['y'] * frame_height))
        pt2 = (int(recorded_data[i]['x'] * frame_width),
               int(recorded_data[i]['y'] * frame_height))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        cv2.circle(frame, pt2, 3, (0, 0, 255), -1)
    return frame


def main():
    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 1080)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    data_collector = DataCollection()
    # initial two algorithm
    pursuits_algorithm = MotionCorrelationEnhanced.PursuitsAlgorithm(
        tau=0.5, N=30)

    path_sync_algorithm = MotionCorrelationEnhanced.PathSyncAlgorithm(
        tau_lo=0.3, tau_hi=0.7, N=30, h=5)

    is_recording = False
    recorded_data = []
    frame_count = 0  #frame count

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame from video stream")
            break
        frame_height, frame_width, _ = frame.shape
        frame, frame_landmarks_data = data_collector.process_frame(
            frame, frame_count)

        if frame_landmarks_data:
            df = pd.DataFrame(frame_landmarks_data)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('r') and not is_recording:
                is_recording = True
                recorded_data = []  # Delete before data
                frame_count = 0  # Reset the counter
                print("Started recording...")

            elif key & 0xFF == ord('s') and is_recording:
                is_recording = False
                print("Stopped recording...")

                recorded_data_df = pd.DataFrame(recorded_data)
                normalized_data = data_collector.normalize_sequence_length(
                    recorded_data_df)

                input_data = normalized_data[['x', 'y']].to_numpy()
                standard_gestures = load_standard_gestures()
                square_data = np.array(standard_gestures['square'])
                circle_data = np.array(standard_gestures['cirle'])
                p_out_list = [square_data, circle_data]

                selected_targets = pursuits_algorithm.run(input_data, p_out_list)
                shape_index = max(set(selected_targets), key=selected_targets.count)
                if shape_index == 0:
                    print("The input shape is square by pursuits.")
                else:
                    print("The input shape is circle.")

                states = path_sync_algorithm.run(input_data, p_out_list)
                activated_counts = np.sum(states == 'Activated', axis=0)
                shape_index = np.argmax(activated_counts)
                if shape_index == 0:
                    print("The input shape is square by path_sync.")
                else:
                    print("The input shape is circle.")

        if is_recording:
            for landmark in frame_landmarks_data:
                if landmark['landmark'] == 8:
                    landmark['time'] = frame_count  # 使用帧计数作为时间标签
                    recorded_data.append(landmark)

            df_record_data = pd.DataFrame(recorded_data)
            df_record_data.to_csv("data/input_data/input.csv", index=False)

            frame = draw_trajectory(
                frame, recorded_data, frame_width, frame_height)

        cv2.imshow('Camera Feed', frame)
        frame_count += 1

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
