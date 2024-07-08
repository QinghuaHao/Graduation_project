from ast import main
import sys
import os
file = os.getcwd()
sys.path.append(file)

from data_collection.data_collection import DataCollection

dc = DataCollection(width=640, height=480, fps=60,
                        duration=10, num_points=80)
print(dc.open_camera())
