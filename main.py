import numpy as np
import pandas as pd
import cv2
import time
import os
import sys
import random
from action_association_analysis.action_association_analysis import MotionCorrelationEnhanced
from data_collection.data_collection import DataCollection
file = os.getcwd()
sys.path.append(file)
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

from LSTM_model.LSTM_CNN import predict,LSTMCNN
def use_model(model_path, df):
    checkpoint = torch.load(model_path)
    model = LSTMCNN(checkpoint['input_size'], checkpoint['hidden_size'], checkpoint['num_layers'],
                    checkpoint['num_classes']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    input_data = preprocess_input_data(df)
    prediction = predict(model, input_data)
    labels_text = ['Circle', 'Square', 'Triangle', 'L Shape']
    print(f"The input trajectory is one{labels_text[prediction]}。")
    return prediction

def preprocess_input_data(df):
    if len(df) < 60:
        raise ValueError(f"Input data size {len(df)} is less than the required size 60")
    data = df[['x', 'y']].values[:60].reshape(1, 60, 2)
    return torch.tensor(data, dtype=torch.float32)

def PreProcessData(df):
    if len(df) < 60:
        while len(df) < 60:
            insert_idx = random.randint(1, len(df) - 1)
            new_row = (df.iloc[insert_idx - 1][['x', 'y']] + df.iloc[insert_idx][['x', 'y']]) / 2
            new_row = pd.Series(new_row, index=df.columns)
            df = pd.concat([df.iloc[:insert_idx], pd.DataFrame([new_row]), df.iloc[insert_idx:]], ignore_index=True)
    elif len(df) > 60:
        excess_count = len(df) - 60
        half_excess = excess_count // 2

        for i in range(half_excess):
            df.iloc[i] = df.iloc[2 * i:(2 * i + 2)].mean()

        for i in range(half_excess, excess_count):
            idx = len(df) - 1 - (excess_count - 1 - i)
            df.iloc[idx] = df.iloc[2 * idx - excess_count:(2 * idx - excess_count + 2)].mean()

        df = df.drop(index=list(range(half_excess)) + list(range(len(df) - excess_count + half_excess, len(df))))
        df = df.reset_index(drop=True)
    return df

def load_standard_gestures():
    gestures = {}
    standard_gestures_dir='data/standard_gestures'
    for file_name in os.listdir(standard_gestures_dir):
        if file_name in ['square.csv', 'cirle.csv']:
            gesture_name = file_name[:-4]
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

def draw_shapes(frame, frame_width, frame_height, active_shape=None):
    shapes = {
        'Circle': ((int(frame_width * 0.2), int(frame_height * 0.8)), 50, (255, 0, 0)),
        'Square': ((int(frame_width * 0.4), int(frame_height * 0.8)), 50, (0, 255, 0)),
        'Triangle': ((int(frame_width * 0.6), int(frame_height * 0.8)), 50, (0, 0, 255)),
        'L Shape': ((int(frame_width * 0.8), int(frame_height * 0.8)), 50, (255, 255, 0))
    }
    for shape, (center, size, color) in shapes.items():
        if shape == active_shape:
            size = int(size * 1.5)
            color = (0, 255, 255)  # Change color for active shape

        if shape == 'Circle':
            cv2.circle(frame, center, size, color, -1)
        elif shape == 'Square':
            top_left = (center[0] - size, center[1] - size)
            bottom_right = (center[0] + size, center[1] + size)
            cv2.rectangle(frame, top_left, bottom_right, color, -1)
        elif shape == 'Triangle':
            points = np.array([[center[0], center[1] - size],
                               [center[0] - size, center[1] + size],
                               [center[0] + size, center[1] + size]])
            cv2.drawContours(frame, [points], 0, color, -1)
        elif shape == 'L Shape':
            pts = np.array([[center[0] - size, center[1] - size],
                            [center[0], center[1] - size],
                            [center[0], center[1]],
                            [center[0] + size, center[1]],
                            [center[0] + size, center[1] + size],
                            [center[0] - size, center[1] + size]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(frame, [pts], color)
    return frame

def main():
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    data_collector = DataCollection()
    pursuits_algorithm = MotionCorrelationEnhanced.PursuitsAlgorithm(tau=0.5, N=30)
    path_sync_algorithm = MotionCorrelationEnhanced.PathSyncAlgorithm(tau_lo=0.3, tau_hi=0.7, N=30, h=5)

    is_recording = False
    recorded_data = []
    frame_count = 0
    active_shape = None
    shape_display_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame from video stream")
            break
        frame_height, frame_width, _ = frame.shape
        frame, frame_landmarks_data = data_collector.process_frame(frame, frame_count)

        if frame_landmarks_data:
            df = pd.DataFrame(frame_landmarks_data)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('r') and not is_recording:
                is_recording = True
                recorded_data = []
                frame_count = 0
                print("Started recording...")

            elif key & 0xFF == ord('s') and is_recording:
                is_recording = False
                print("Stopped recording...")

                recorded_data_df = pd.DataFrame(recorded_data)
                normalized_data = PreProcessData(recorded_data_df)

                input_data = normalized_data[['x', 'y']].to_numpy()

                prediction = use_model('motion_complex_lstm_cnn_model_tuned.pth', normalized_data)
                active_shape = ['Circle', 'Square', 'Triangle', 'L Shape'][prediction]
                shape_display_time = time.time()

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
                    landmark['time'] = frame_count
                    recorded_data.append(landmark)

            df_record_data = pd.DataFrame(recorded_data)
            frame = draw_trajectory(frame, recorded_data, frame_width, frame_height)

        if active_shape and time.time() - shape_display_time < 2:
            frame = draw_shapes(frame, frame_width, frame_height, active_shape)
        else:
            frame = draw_shapes(frame, frame_width, frame_height)

        cv2.imshow('Camera Feed', frame)
        frame_count += 1

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
