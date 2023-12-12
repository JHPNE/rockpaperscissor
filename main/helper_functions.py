import time
import numpy as np

REFERENCE_VECTORS = {
    'up': np.array([0, -1, 0]),
    'down': np.array([0, 1, 0]),
    'right': np.array([1, 0, 0]),
    'left': np.array([-1, 0, 0]),
    'forward': np.array([0, 0, -1]),
    'backward': np.array([0, 0, 1])
}

def fps_calculator(previous_time):
    current_time = time.time()
    fps = 1/ (current_time - previous_time)
    return fps, current_time

def move_detection(hand_landmarkers, orientation):
    if hand_landmarkers is not None:
        landmarks = hand_landmarkers.landmark

        wrist = [landmarks[0].x, landmarks[0].y, 0] 

        pinky_tip = [landmarks[20].x, landmarks[20].y, 0]
        pinky_mcp = [landmarks[17].x, landmarks[17].y, 0]

        ring_finger_tip = [landmarks[16].x, landmarks[16].y, 0]
        ring_finger_mcp = [landmarks[13].x, landmarks[13].y, 0]

        middle_finger_tip = [landmarks[12].x, landmarks[12].y, 0]
        middle_finger_mcp = [landmarks[9].x, landmarks[9].y, 0]

        index_finger_tip = [landmarks[8].x, landmarks[8].y, 0]
        index_finger_mcp = [landmarks[5].x, landmarks[5].y, 0]

        pinky_diff = vector_length(pinky_mcp, wrist) > vector_length(pinky_tip, wrist)
        ring_finger_diff = vector_length(ring_finger_mcp, wrist) > vector_length(ring_finger_tip, wrist)
        middle_finger_diff = vector_length(middle_finger_mcp, wrist) > vector_length(middle_finger_tip, wrist)
        index_finger_diff = vector_length(index_finger_mcp, wrist) > vector_length(index_finger_tip, wrist)
       
        ring_finger_pip = [landmarks[14].x, landmarks[14].y, 0]
        ring_finger_pip_diff = vector_length(ring_finger_pip, wrist) > vector_length(ring_finger_tip, wrist) 

        if all([pinky_diff, ring_finger_diff, middle_finger_diff, index_finger_diff]):
            return 'rock' 
        elif all([pinky_diff, ring_finger_diff, ring_finger_pip_diff]):
            return 'scissor'
        else:
            return 'hand' 


# counter movement 
def counter():
    pass

# brow indicator
def brow_movement(face_landmarks):
    if face_landmarks is not None:
        landmarks = face_landmarks.landmark

    pass

# mouth indicator
def mouth_movement(face_landmarks):
    if face_landmarks is not None:
        landmarks = face_landmarks.landmark

        mouth_right_top = [landmarks[37].x, landmarks[37].y, 0]
        mouth_right_bottom = [landmarks[84].x, landmarks[84].y, 0]
        mouth_mid_top = [landmarks[0].x, landmarks[0].y, 0]
        mouth_mid_bottom = [landmarks[17].x, landmarks[17].y, 0]
        mouth_left_top = [landmarks[267].x, landmarks[267].y, 0]
        mouth_left_bottom = [landmarks[314].x, landmarks[314].y, 0]

        ind_1 = vector_length(mouth_right_top, mouth_right_bottom)
        ind_2 = vector_length(mouth_mid_top, mouth_mid_bottom)
        ind_3 = vector_length(mouth_left_top, mouth_left_bottom)

        threshold = 0.08

        # twitching mouth corner
        mouth_corner_left = [landmarks[287].x, landmarks[287].y, 0]
        mouth_corner_right = [landmarks[57].x, landmarks[57].y, 0]

        mouth_peak_left = [landmarks[436].x, landmarks[436].y, 0]
        mouth_peak_right = [landmarks[436].x, landmarks[436].y, 0]

        ind_4 = vector_length(mouth_corner_left, mouth_peak_left)
        ind_5 = vector_length(mouth_corner_right, mouth_peak_right)

        threshold_twitch = 0.038

        if all([ round(i, 4) <= threshold for i in [ind_1, ind_2, ind_3]]):
            return 'clutching'
        else:
            return 'open'

# eye indicator
def eye_movement(face_landmarks):
    if face_landmarks is not None:
        landmarks = face_landmarks.landmark

        left_eye_3_top = [landmarks[160].x, landmarks[160].y, 0]
        left_eye_3_bottom = [landmarks[144].x, landmarks[144].y, 0]
        left_eye_4_top = [landmarks[159].x, landmarks[159].y, 0]
        left_eye_4_bottom = [landmarks[145].x, landmarks[145].y, 0]
        left_eye_5_top = [landmarks[158].x, landmarks[158].y, 0]
        left_eye_5_bottom = [landmarks[153].x, landmarks[153].y, 0]

        ind_3 = vector_length(left_eye_3_top, left_eye_3_bottom)
        ind_4 = vector_length(left_eye_4_top, left_eye_4_bottom)
        ind_5 = vector_length(left_eye_5_top, left_eye_5_bottom)

        threshold = 0.018

        if all([ round(i, 4) <= threshold for i in [ind_3, ind_4, ind_5]]):
            return 'closing'
        else:
            return 'open'

# clutched indicator
def clutched_or_relaxed(hand_landmarks):
    if hand_landmarks is not None:
        landmarks = hand_landmarks.landmark

        pinky_tip = [landmarks[20].x, landmarks[20].y, 0]
        ring_finger_tip = [landmarks[16].x, landmarks[16].y, 0]
        middle_finger_tip = [landmarks[12].x, landmarks[12].y, 0]
        index_finger_tip = [landmarks[8].x, landmarks[8].y, 0]

        index_middle_finger = vector_length(index_finger_tip, middle_finger_tip)
        middle_ring_finger = vector_length(middle_finger_tip, ring_finger_tip)
        pinky_ring = vector_length(ring_finger_tip, pinky_tip)

        min_threshold = 0.03
        max_threshold = 0.06

        print(index_middle_finger)

        if all([ round(i,3) <= min_threshold for i in [index_middle_finger, middle_ring_finger, pinky_ring]]):
            return 'clutched'
        elif all([ round(i,3) >= max_threshold for i in [index_middle_finger, middle_ring_finger, pinky_ring]]):
            return 'relaxed'

def vector_length(a, b):
    vector = np.array(a) - np.array(b)
    return np.sqrt(vector.dot(vector))

def get_orientation(hand_landmarkers):
    if hand_landmarkers is not None:
        landmarks = hand_landmarkers.landmark
        wrist = [landmarks[0].x, landmarks[0].y, landmarks[0].z]
        fingertip_left = [landmarks[5].x, landmarks[5].y, landmarks[5].z]
        fingertip_right = [landmarks[17].x, landmarks[17].y, landmarks[17].z]

        vector_left = np.array(fingertip_left) - np.array(wrist) 
        vector_right = np.array(fingertip_right) - np.array(wrist) 

        normal_vector = np.cross(vector_left, vector_right) 

        angles = {}

        for direction, ref_vector in REFERENCE_VECTORS.items():
            dot_product = np.dot(normal_vector, ref_vector)
            normal_mag = np.linalg.norm(normal_vector)
            ref_mag = np.linalg.norm(ref_vector)
            angle_rad = np.arccos(dot_product / (normal_mag * ref_mag))
            angle_deg = np.degrees(angle_rad)
            angles[direction] = angle_deg

        leaning_directions = [direction for direction, angle in angles.items() if angle < 90]

        return leaning_directions
