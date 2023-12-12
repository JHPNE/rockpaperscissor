import cv2
import mediapipe as mp

from helper_functions import fps_calculator, move_detection, get_orientation, clutched_or_relaxed, eye_movement, mouth_movement 

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# For webcam input:
cap = cv2.VideoCapture(0)
previous_time = 0

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
    mp_drawing.draw_landmarks(
      image,
      results.face_landmarks,
      mp_holistic.FACEMESH_CONTOURS,
      mp_drawing.DrawingSpec(
        color=(255,0,255),
        thickness=1,
        circle_radius=1
      ),
      mp_drawing.DrawingSpec(
        color=(0,255,255),
        thickness=1,
        circle_radius=1
      )
    )
    
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
    
    value = get_orientation(results.right_hand_landmarks)
    move = move_detection(results.right_hand_landmarks, value)
    pred = clutched_or_relaxed(results.right_hand_landmarks)
    eye = eye_movement(results.face_landmarks)
    mouth = mouth_movement(results.face_landmarks)

    fps, current_time = fps_calculator(previous_time)
    previous_time = current_time

    frame = cv2.flip(image, 1)

    cv2.putText(frame, str(move)+" MOVE", (5, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2) 
    cv2.putText(frame, str(mouth)+" ORIENTATION", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2) 
    cv2.putText(frame, str(int(fps))+" FPS", (15, 105), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2) 

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('frame', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()

