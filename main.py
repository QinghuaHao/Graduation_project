from ast import main
import sys
import os
file = os.getcwd()
sys.path.append(file)

from data_collection.data_collection import DataCollection
import cv2
import pandas as pd
import numpy as np


'''
1.数据流
2.if 手势ok 收集数据流,直到victory结束,返回数据包
3.根据数据包形成标准轨迹
4.进行匹配
5,匹配完成以后输出
'''

dc = DataCollection()
cap =cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera can not open")
else:
    recording = False
    collected_data = []
    i=0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can not receive frame")
            break
        
        frame = cv2.flip(frame, 1)  # 正确翻转镜头
        
        for df, gesture in dc.process_video(cap):
            print(f"gesture_is: {gesture}")
            # print("data_frame_is")
            # print(df)
            
            if gesture == 'Thumbs Up':
                recording = True
                collected_data = []  # 开始记录时清空之前的记录数据

            if recording:
                collected_data.append(df)  # 记录当前帧的特征点数据
            
            if gesture == 'Thumbs Down' and recording:
                recording = False
                i+=1
                # 将收集到的数据组合成一个 DataFrame
                full_data = pd.concat(collected_data, ignore_index=True)
                print(f"Collected data{i}:")
                print(full_data)
                collected_data = []
            


        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cap.destroyAllWindows('Camera Feed')