import cv2
import mediapipe as mp

class dataCollection():

	def __init__(self):
		self.mp_hands = mp.solutions.hands
		self.hands = self.mp_hands.Hands()
		self.mp_drawing =mp.solutions.drawing_utils
		self.landmarks_data = []


	def open_camera(self):
		cap = cv2.VideoCapture(0)
		if not cap.isOpened():
			print("can not open camera")
			return
		frame_count = 0
		while True:
			ret, frame = cap.read()
			if not ret:
				print("camera can not open")
				break
			frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
			results = self.hands.process(frame)
			frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
			if results.multi_hand_landmarks:
				for hand_id,hand_landmarks in enumerate(results.multi_hand_landmarks):
					self.mp_drawing.draw_landmarks(frame,hand_landmarks,self.mp_hands.HAND_CONNECTIONS)
					for landmark_id, landmark in enumerate(hand_landmarks.landmark):
						landmark_data={
						'frame':frame_count,
						'hand':hand_id,
						'landmark':landmark_id,
						'x':landmark.x,
						'y':landmark.y,
						'z':landmark.z
						}
						self.landmarks_data.append(landmark_data)
						yield landmark_data
			cv2.imshow('Camera Feed', frame)
			# print(self.landmarks_data)
			if cv2.waitKey(1)&0xFF == ord('q'):
				break
			frame_count+=1
		cap.release()
		cv2.destroyWindow('Camera Feed')

if __name__ =='__main__':
	dc = dataCollection()
	for landmark in dc.open_camera():
		print(landmark)