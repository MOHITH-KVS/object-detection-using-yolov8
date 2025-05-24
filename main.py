import cv2
import mediapipe as mp
import time
from collections import deque
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

last_object = None
start_time = None
hold_duration = 0
buffer_size = 10
detection_buffer = deque(maxlen=buffer_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_present = False
    object_in_hand = None

    # Run YOLO on the full frame
    yolo_results = model(frame)
    detected_boxes = []

    for result in yolo_results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_boxes.append((label, (x1, y1, x2, y2)))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if results.multi_hand_landmarks:
        hand_present = True
        # Get hand bounding box
        landmarks = results.multi_hand_landmarks[0].landmark
        x_coords = [int(lm.x * w) for lm in landmarks]
        y_coords = [int(lm.y * h) for lm in landmarks]
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)
        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        # Check for any object (except 'person') overlapping with hand
        hx1, hy1, hx2, hy2 = xmin, ymin, xmax, ymax
        for label, (cx1, cy1, cx2, cy2) in detected_boxes:
            if label.lower() == "person":
                continue  # Ignore 'person' class
            if (hx1 < cx2 and hx2 > cx1 and hy1 < cy2 and hy2 > cy1):
                object_in_hand = label
                break  # Take the first valid overlapping object

    # Buffer logic for smoothing
    detection_buffer.append(object_in_hand if hand_present else None)
    # Find the most frequent non-None object in the buffer
    if hand_present:
        objects = [obj for obj in detection_buffer if obj is not None]
        if objects:
            stable_object = max(set(objects), key=objects.count)
            if last_object != stable_object:
                start_time = time.time()
                last_object = stable_object
            hold_duration = int(time.time() - start_time)
            message = f"{stable_object.capitalize()} detected in your hand ({hold_duration}s)"
        else:
            last_object = None
            hold_duration = 0
            message = "No object detected in your hand"
    else:
        last_object = None
        hold_duration = 0
        message = "Show your hand to start detection"

    cv2.putText(frame, message, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(frame, "Press 'q' to Quit", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Any Object-in-Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
