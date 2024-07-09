from action_association_analysis.action_association_analysis import MotionCorrelationEnhanced
import numpy as np
import pandas as pd
import cv2
from data_collection.data_collection import DataCollection
import time

import sys
import os
file = os.getcwd()
sys.path.append(file)


'''
1.数据流
2.if 手势ok 收集数据流,直到victory结束,返回数据包
3.根据数据包形成标准轨迹
4.进行匹配
5,匹配完成以后输出
'''


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
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    data_collector = DataCollection()
    # 初始化两个算法
    pursuits_algorithm = MotionCorrelationEnhanced.PursuitsAlgorithm(
        tau=0.5, N=30)
    
    path_sync_algorithm = MotionCorrelationEnhanced.PathSyncAlgorithm(
        tau_lo=0.3, tau_hi=0.7, N=30, h=5)

    is_recording = False
    recorded_data = []
    frame_count = 0  # 每一帧的计数器
    start_time = time.time()
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
            # gesture = data_collector.detect_gesture(df)
            # print(gesture)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('r') and not is_recording:
                is_recording = True
                recorded_data = []  # 开始记录时清空之前的数据
                print("Started recording...")

            elif key & 0xFF == ord('s') and is_recording:
                is_recording = False
                print("Stopped recording...")

                recorded_data_df = pd.DataFrame(recorded_data)
                normalized_data = data_collector.normalize_sequence_length(
                    recorded_data_df)

                input_data = normalized_data[['x', 'y']].to_numpy()
                standard_gestures = data_collector.load_standard_gestures()
                # print('-----------+++++++')
                # print(standard_gestures)
                square_data = np.array(standard_gestures['square'])
                circle_data = np.array(standard_gestures['cirle'])
                # print(square_data)
                # print(circle_data)
                p_out_list = [square_data,circle_data]
                selected_targets = pursuits_algorithm.run(input_data, p_out_list)
                shape_index = max(set(selected_targets), key=selected_targets.count)
                if shape_index == 0:
                    print("The input shape is square by pursuits.")
                else:
                    print("The input shape is circle.")

                states = path_sync_algorithm.run(input_data, p_out_list)
                # 判断状态结果
                activated_counts = np.sum(states == 'Activated', axis=0)
                shape_index = np.argmax(activated_counts)
                if shape_index == 0:
                    print("The input shape is square by path_sync.")


                    display_duration = 5
                    current_time = time.time()
                    if current_time - start_time < display_duration:
                        text = "OpenCV Text"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        position = (50, 50)  # 文本位置
                        font_scale = 1
                        font_color = (255, 255, 255)  # 白色
                        line_type = 2
                        cv2.putText(frame, text, position, font, font_scale, font_color, line_type)


                else:
                    print("The input shape is circle.")

        if is_recording:
            for landmark in frame_landmarks_data:
                if landmark['landmark'] == 8:
                    landmark['frame'] = frame_count
                    recorded_data.append(landmark)
            # print("------------")
            # print(recorded_data)
            frame = draw_trajectory(frame, recorded_data, frame_width, frame_height)

        cv2.imshow('Camera Feed', frame)
        frame_count += 1

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
