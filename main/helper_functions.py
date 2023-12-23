import time
import numpy as np
import cv2

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
            return 'Rock' 
        elif all([pinky_diff, ring_finger_diff, ring_finger_pip_diff]):
            return 'Scissors'
        else:
            return 'Paper' 

def add_text_center(frame, move_text, elapsed_time_text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    move_size = cv2.getTextSize(move_text, font, 1.5, 3)[0]  
    time_size = cv2.getTextSize(elapsed_time_text, font, 5, 7)[0]  

    move_x = (frame.shape[1] - move_size[0]) // 2
    move_y = (frame.shape[0] - move_size[1]) // 2

    time_x = (frame.shape[1] - time_size[0]) // 2
    time_y = move_y + move_size[1] + 100 

    if int(elapsed_time_text) > 1:
        cv2.putText(frame, move_text, (move_x, move_y), font, 1.5, (255, 0, 0), 3, cv2.LINE_AA)  
        cv2.putText(frame, elapsed_time_text, (time_x, time_y), font, 5, (255, 0, 0), 7, cv2.LINE_AA)  

def countdown_timer(countdown):
    time.sleep(1)
    return countdown - 1

def resultsAsd(player1_move, player2_move):
    win_conditions = {
        'Rock': 'Scissors',
        'Paper': 'Rock',
        'Scissors': 'Paper'
    }

    if player1_move == player2_move:
        return "Tie"
    elif win_conditions[player1_move] == player2_move:
        return "Win"
    else:
        return "Lose"


def win_lose_trade(games_info):
    moves_dict = {
        "Rock": "Paper",
        "Paper": "Scissors",
        "Scissors": "Rock"
    }

    move  = games_info['move']
    result = games_info['result']

    if result == 'win':
        return moves_dict.get(move)
    else:
        return move

# counter movement 
def counter(mouth_info, eye_info, brow_info, hand_info):
    hand_indication = determine_rock_paper_scissors_mouth_eye_hand(hand_info)
    mouth_indication = determine_rock_paper_scissors_mouth_eye_hand(mouth_info)
    eye_indication = determine_rock_paper_scissors_mouth_eye_hand(eye_info)
    brow_indication = determine_rock_paper_scissors_mouth_eye_hand(brow_info)

    indicators = [hand_indication, mouth_indication, eye_indication, brow_indication]
    return indicators

def evaluate_moves_initial(indicators):
    counts = {'Rock': 0, 'Paper': 0, 'Scissors': 0}
    weights = {'hand': 1, 'mouth': 2, 'eye': 0.5, 'brow': 0.5}

    [hand, mouth, eye, brow] = indicators

    counts[hand] += weights['hand']
    counts[mouth] += weights['mouth']
    counts[eye] += weights['eye']
    counts[brow] += weights['brow']
    
    v = list(counts.values())
    k = list(counts.keys())

    return k[v.index(max(v))]

def evaluate_move(indicators, games_info):
    evaluated_move = evaluate_moves_initial(indicators)
    win_lose_move = win_lose_trade(games_info)

    if win_lose_move == evaluated_move:
        return win_lose_move
    else: 
        return win_lose_move

def determine_eyebrow_movement(face_landmarks, threshold=0.1):
    if face_landmarks is not None:
        landmarks = face_landmarks.landmark

        left_eyebrow_inner = [landmarks[276].x, landmarks[276].y, 0]
        left_eyebrow_outer = [landmarks[283].x, landmarks[283].y, 0]  
        right_eyebrow_inner = [landmarks[324].x, landmarks[324].y, 0]  
        right_eyebrow_outer = [landmarks[321].x, landmarks[321].y, 0] 

        diff_left_x = left_eyebrow_inner[0] - left_eyebrow_outer[0]
        diff_left_y = left_eyebrow_inner[1] - left_eyebrow_outer[1]
        diff_right_x = right_eyebrow_outer[0] - right_eyebrow_inner[0]
        diff_right_y = right_eyebrow_outer[1] - right_eyebrow_inner[1]

        if abs(diff_left_x) < threshold and abs(diff_left_y) < threshold:
            left_movement = "Still"
        elif diff_left_x > threshold and diff_left_y > threshold:
            left_movement = "Moved up and left"
        elif diff_left_x > threshold and diff_left_y < -threshold:
            left_movement = "Moved down and left"
        elif diff_left_x < -threshold and diff_left_y > threshold:
            left_movement = "Moved up and right"
        elif diff_left_x < -threshold and diff_left_y < -threshold:
            left_movement = "Moved down and right"
        else:
            left_movement = "Undefined"

        if abs(diff_right_x) < threshold and abs(diff_right_y) < threshold:
            right_movement = "Still"
        elif diff_right_x > threshold and diff_right_y > threshold:
            right_movement = "Moved up and left"
        elif diff_right_x > threshold and diff_right_y < -threshold:
            right_movement = "Moved down and left"
        elif diff_right_x < -threshold and diff_right_y > threshold:
            right_movement = "Moved up and right"
        elif diff_right_x < -threshold and diff_right_y < -threshold:
            right_movement = "Moved down and right"
        else:
            right_movement = "Undefined"

        return left_movement, right_movement

def determine_rock_paper_scissors_brow(left_movement, right_movement):
    if left_movement == "Moved down and left" or right_movement == "Moved down and left":
        return "Rock"
    elif left_movement == "Moved up and left" or right_movement == "Moved up and left":
        return "Paper"
    elif (
        left_movement == "Moved up and right"
        or right_movement == "Moved up and right"
        or left_movement == "Moved down and right"
        or right_movement == "Moved down and right"
    ):
        return "Scissors"
    else:
        return "Undefined"

def determine_rock_paper_scissors_mouth_eye_hand(states):
    open = 0
    close = 0
    max_duration_open = 0
    max_duration_close = 0
    current_max_duration = 0
    current_state = None

    for state in states:
        if state == 'open':
            open += 1

            if current_state != 'open':
                current_state = 'open'
                current_max_duration = 0
                max_duration_open = max(max_duration_open, current_max_duration)
            else:
                current_max_duration += 1
        else:
            close += 1
            
            if current_state != 'close':
                current_state = 'close'
                current_max_duration = 0
                max_duration_open = max(max_duration_close, current_max_duration)
            else:
                current_max_duration += 1

    if max_duration_open > max_duration_close and open > close:
        return "Paper"
    elif max_duration_open < max_duration_close and open <= close:
        return "Rock"
    else:
        return "Scissors"

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
            return 'close'
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
            return 'close'
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

        if all([ round(i,3) <= min_threshold for i in [index_middle_finger, middle_ring_finger, pinky_ring]]):
            return 'close'
        elif all([ round(i,3) >= max_threshold for i in [index_middle_finger, middle_ring_finger, pinky_ring]]):
            return 'open'

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
