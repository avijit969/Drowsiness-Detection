import cv2
import mediapipe as mp
import serial
import time

# -------------------- Arduino Setup --------------------
# arduino = serial.Serial('COM5', 9600)
# time.sleep(2) 

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def get_eye_aspect_ratio(landmarks, eye_indices):
    """
    Calculates the Eye Aspect Ratio (EAR)
    using 6 landmarks of the eye.
    """
    left = landmarks[eye_indices[0]]
    right = landmarks[eye_indices[3]]
    top = (landmarks[eye_indices[1]][1] + landmarks[eye_indices[2]][1]) / 2
    bottom = (landmarks[eye_indices[4]][1] + landmarks[eye_indices[5]][1]) / 2
    ear = abs(bottom - top) / abs(right[0] - left[0])
    return ear

# -------------------- Eye Landmark Indices --------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# -------------------- Video Capture --------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

closed_frames = 0
threshold_frames = 20  
ear_threshold = 0.2500

print("✅ Drowsiness detector started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=[],
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            )

            for idx in LEFT_EYE + RIGHT_EYE:
                (x, y) = landmarks[idx]
                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)

            left_ear = get_eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = get_eye_aspect_ratio(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2

            cv2.putText(frame, f"EAR: {avg_ear:.4f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Determine drowsiness
            if avg_ear < ear_threshold:
                closed_frames += 1
                cv2.putText(frame, "Eyes Closed", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                closed_frames = 0
                cv2.putText(frame, "Eyes Open", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # arduino.write(b'0')  # Turn OFF buzzer

            # Trigger buzzer if drowsy
            if closed_frames > threshold_frames:
                cv2.putText(frame, "⚠️ DROWSINESS DETECTED ⚠️", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                print("Drowsiness detected!")
                # arduino.write(b'1')  # Turn ON buzzer
                closed_frames = 0

    # Show the frame
    cv2.imshow("Driver Drowsiness Detector", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# arduino.close()
cv2.destroyAllWindows()
