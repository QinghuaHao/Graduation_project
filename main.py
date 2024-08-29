import cv2
import numpy as np
import pandas as pd
import time
import torch
import random
import os
import sys
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

from action_association_analysis.action_association_analysis import MotionCorrelationEnhanced
from data_collection.data_collection import DataCollection
from LSTM_model.LSTM_CNN import predict,LSTMCNN
def use_model(model_path, df):
    checkpoint = torch.load(model_path)
    model = LSTMCNN(checkpoint['input_size'], checkpoint['hidden_size'], checkpoint['num_layers'], checkpoint['num_classes']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    input_data = preprocess_input_data(df)
    prediction = predict(model, input_data)

    labels_text = ['Circle', 'Square', 'Triangle', 'L Shape']
    print(f"The trajectory is one {labels_text[prediction]}.")
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

def draw_shapes(frame, frame_width, frame_height, active_shape=None, animation_progress=0):
    shapes = {
        'Circle': ((int(frame_width * 0.2), int(frame_height * 0.8)), 50, (255, 0, 0)),
        'Square': ((int(frame_width * 0.4), int(frame_height * 0.8)), 50, (0, 255, 0)),
        'Triangle': ((int(frame_width * 0.6), int(frame_height * 0.8)), 50, (0, 0, 255)),
        'L': ((int(frame_width * 0.8), int(frame_height * 0.8)), 50, (255, 255, 0))
    }

    glow_color = (255, 255, 255)  # White color for the glow effect
    glow_thickness = 10  # Thickness of the glow effect
    outline_color = (0, 0, 0)  # Black color for the outline
    outline_thickness = 2

    for shape, (center, size, color) in shapes.items():
        if shape == active_shape:
            size = int(size * 1.5)  # Apply size increase for the active shape
            if shape == 'L':
                outline_color = (0, 255, 255)  # Change outline color specifically for active L shape
            else:
                color = (0, 255, 255)  # Change color for other active shapes

        # Draw the shape
        if shape == 'Circle':
            cv2.circle(frame, center, size, color, -1)

            # Outline the circle with a dynamic border starting from top-left (45 degrees)
            start_angle = -45  # Start at top-left corner
            angle_extent = int(360 * animation_progress)
            cv2.ellipse(frame, center, (size + glow_thickness, size + glow_thickness), 0, start_angle, start_angle + angle_extent, glow_color, glow_thickness)
            cv2.circle(frame, center, size, outline_color, outline_thickness)

        elif shape == 'Square':
            top_left = (center[0] - size, center[1] - size)
            bottom_right = (center[0] + size, center[1] + size)
            cv2.rectangle(frame, top_left, bottom_right, color, -1)

            # Dynamic outline
            total_perimeter = 4 * size * 2
            current_length = int(total_perimeter * animation_progress)

            # Top border
            if current_length <= 2 * size:
                end_point = top_left[0] + current_length
                cv2.line(frame, top_left, (end_point, top_left[1]), glow_color, glow_thickness)
            else:
                cv2.line(frame, top_left, (bottom_right[0], top_left[1]), glow_color, glow_thickness)
                remaining = current_length - 2 * size

                # Right border
                if remaining <= 2 * size:
                    end_point = top_left[1] + remaining
                    cv2.line(frame, (bottom_right[0], top_left[1]), (bottom_right[0], end_point), glow_color, glow_thickness)
                else:
                    cv2.line(frame, (bottom_right[0], top_left[1]), bottom_right, glow_color, glow_thickness)
                    remaining -= 2 * size

                    # Bottom border
                    if remaining <= 2 * size:
                        end_point = bottom_right[0] - remaining
                        cv2.line(frame, bottom_right, (end_point, bottom_right[1]), glow_color, glow_thickness)
                    else:
                        cv2.line(frame, bottom_right, (top_left[0], bottom_right[1]), glow_color, glow_thickness)
                        remaining -= 2 * size

                        # Left border
                        if remaining > 0:
                            end_point = bottom_right[1] - remaining
                            cv2.line(frame, (top_left[0], bottom_right[1]), (top_left[0], end_point), glow_color, glow_thickness)

            cv2.rectangle(frame, top_left, bottom_right, outline_color, outline_thickness)

        elif shape == 'Triangle':
            points = np.array([[center[0], center[1] - size],
                               [center[0] - size, center[1] + size],
                               [center[0] + size, center[1] + size]])

            cv2.drawContours(frame, [points], 0, color, -1)

            # Dynamic outline for triangle
            perimeter = size * 3
            current_length = perimeter * animation_progress

            if current_length <= size:
                # First side (top to bottom-left)
                end_point = points[0] + (points[1] - points[0]) * (current_length / size)
                cv2.line(frame, points[0], end_point.astype(int), glow_color, glow_thickness)
            elif current_length <= 2 * size:
                # First side complete, start the second side (bottom-left to bottom-right)
                cv2.line(frame, points[0], points[1], glow_color, glow_thickness)
                remaining_length = current_length - size
                end_point = points[1] + (points[2] - points[1]) * (remaining_length / size)
                cv2.line(frame, points[1], end_point.astype(int), glow_color, glow_thickness)
            else:
                # First two sides complete, finish with the third side (bottom-right to top)
                cv2.line(frame, points[0], points[1], glow_color, glow_thickness)
                cv2.line(frame, points[1], points[2], glow_color, glow_thickness)
                remaining_length = current_length - 2 * size
                end_point = points[2] + (points[0] - points[2]) * (remaining_length / size)
                cv2.line(frame, points[2], end_point.astype(int), glow_color, glow_thickness)

            cv2.drawContours(frame, [points], 0, outline_color, outline_thickness)

        elif shape == 'L':
            half_size = size // 1.1
            pts = np.array([[center[0] - size, center[1] - size],  # Top left
                            [center[0] - size, center[1] + size],  # Bottom left
                            [center[0] - size + half_size, center[1] + size]], np.int32)  # Bottom right (half length)
            pts = pts.reshape((-1, 1, 2))

            # Draw L shape with the size adjusted for active shape
            cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=glow_thickness)

            # Dynamic outline for "L" shape
            total_length = 2 * size + half_size  # Vertical + Half Horizontal
            current_length = total_length * animation_progress

            if current_length <= 2 * size:
                # Draw the vertical part
                end_point_y = pts[0][0][1] + int(current_length)
                cv2.line(frame, (pts[0][0][0], pts[0][0][1]), (pts[0][0][0], end_point_y), glow_color, glow_thickness)
            else:
                # Vertical part complete, start the horizontal part
                cv2.line(frame, pts[0][0], pts[1][0], glow_color, glow_thickness)
                remaining_length = current_length - 2 * size

                if remaining_length > 0:
                    end_point_x = pts[1][0][0] + int(remaining_length)
                    cv2.line(frame, pts[1][0], (end_point_x, pts[1][0][1]), glow_color, glow_thickness)

            # Draw outline
            cv2.polylines(frame, [pts], isClosed=False, color=outline_color, thickness=outline_thickness)

    return frame

def main():
    cap = cv2.VideoCapture(0)

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
    start_time = time.time()
    animation_duration = 2  # Duration of the animation in seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame from video stream")
            break
        frame_height, frame_width, _ = frame.shape
        frame, frame_landmarks_data = data_collector.process_frame(frame, frame_count)
        # Calculate animation progress (looping from 0 to 1)
        elapsed_time = time.time() - start_time
        animation_progress = (elapsed_time % animation_duration) / animation_duration
        df = pd.DataFrame(frame_landmarks_data)

        # Detect gesture
        detected_gesture = data_collector.detect_gesture(df)
        print(f"Detected Gesture: {detected_gesture}")

        if frame_landmarks_data:
            key = cv2.waitKey(1)
            if detected_gesture == 'Index Finger Up' and not is_recording:
                is_recording = True
                recorded_data = []
                frame_counqt = 0
                print("Started recording...")

            elif detected_gesture == 'Fist' and is_recording:
                is_recording = False
                print("Stopped recording...")

                recorded_data_df = pd.DataFrame(recorded_data)
                normalized_data = PreProcessData(recorded_data_df)

                input_data = normalized_data[['x', 'y']].to_numpy()

                prediction = use_model('LSTM_model/Saved_Model/motion_lstm_cnn_model_tuned.pth', normalized_data)
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
                if landmark['landmark'] == 8:  # Assuming landmark 8 is the point of interest
                    landmark['time'] = frame_count
                    recorded_data.append(landmark)

            df_record_data = pd.DataFrame(recorded_data)
            frame = draw_trajectory(frame, recorded_data, frame_width, frame_height)

        if active_shape and time.time() - shape_display_time < 2:
            frame = draw_shapes(frame, frame_width, frame_height, active_shape, animation_progress)
        else:
            frame = draw_shapes(frame, frame_width, frame_height, animation_progress=animation_progress)

        cv2.imshow('Camera Feed', frame)
        frame_count += 1

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
