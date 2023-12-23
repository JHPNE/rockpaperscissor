import cv2
import mediapipe as mp
import time
import threading

from helper_functions import *
# from helper_functions import fps_calculator, move_detection, get_orientation, clutched_or_relaxed, eye_movement, mouth_movement, countdown_timer, add_text_center, get_counter_part, determine_eyebrow_movement, counter, evaluate_move, resultsAsd, evaluate_moves_initial

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

initial_countdown = 6

# For webcam input:
cap = cv2.VideoCapture(0)
previous_time = 0
countdown = initial_countdown
start_time = time.time()

hand_info = []
eye_info = []
mouth_info = []
brow_info = []
moves_info = []

rounds = 0
result = ''
games_info = {
    'move': None,
    'result': None,
}
indicator_move = "Predicting..."

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
         # If loading a video, use 'break' instead of 'continue'.
        break 

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Draw Face Landmarks
#
#    mp_drawing.draw_landmarks(
#      image,
#      results.face_landmarks,
#      mp_holistic.FACEMESH_CONTOURS,
#    )
    
    # Draw Hand Landmarks
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )

    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )

    fps, current_time = fps_calculator(previous_time)
    previous_time = current_time

    frame = cv2.flip(image, 1)

    elapsed_time_second = int(time.time() - start_time)
    time_diff = countdown - elapsed_time_second
    if time_diff == initial_countdown - 3:
        indicator_move = "Predicting..."

    if time_diff > 0:
        add_text_center(frame, "Chose your weapon (right hand only)", str(time_diff))

        value = get_orientation(results.right_hand_landmarks)
        move = move_detection(results.right_hand_landmarks, value)
        hand = clutched_or_relaxed(results.right_hand_landmarks)
        eye = eye_movement(results.face_landmarks)
        mouth = mouth_movement(results.face_landmarks)
        brow = determine_eyebrow_movement(results.face_landmarks)

        hand_info.append(hand)
        eye_info.append(eye)
        mouth_info.append(mouth)
        brow_info.append(brow)
        moves_info.append(move)


    if time_diff == 0 and len(moves_info) > 0:
        indicators = counter(mouth_info, eye_info, brow_info, hand_info)
        move = moves_info[-1]
        result = ''

        if move == None:
             start_time = time.time()
             rounds += 1
             countdown = initial_countdown
             continue

        indicator_move = get_counter_part(evaluate_moves_initial(indicators) if rounds == 0 else evaluate_move(indicators, games_info))

        result = resultsAsd(move, indicator_move)
        games_info['move'] = move 
        games_info['result'] = result

        countdown = initial_countdown
        start_time = time.time()
        rounds += 1
        
        hand_info = []
        eye_info = []
        mouth_info = []
        brow_info = []
        moves_info = []

    cv2.putText(frame, 'Actual player move: ' + str(move), (35, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3, cv2.LINE_AA)  
    cv2.putText(frame, 'Computer played: ' + str(indicator_move), (35, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3, cv2.LINE_AA)  

    cv2.imshow('frame', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()

