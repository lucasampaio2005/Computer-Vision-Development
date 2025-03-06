import cv2
from ultralytics import YOLO
import numpy as np

vide_filename = "data/1.mp4"

model = YOLO('yolov8n-pose.pt')

cap = cv2.VideoCapture(vide_filename)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_filename = 'output_with_result.avi'
writter = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

JUMP_THRESHOLD = 0.1
ROTATION_THRESHOLD = 2
CONF_THRESHOLD = 0.5

prev_keypoints = None
movement_state = 'Hareketsiz'
frame_count = 0

previous_states = []
stability_frames = 5

def analyze_movement(curr_keypoints, prev_keypoints):
    
    if prev_keypoints is None:
        return "Hareketsiz"
    
    if(curr_keypoints[15, 2] < CONF_THRESHOLD or curr_keypoints[16, 2] < CONF_THRESHOLD or curr_keypoints[5, 2] < CONF_THRESHOLD or curr_keypoints[6, 2] < CONF_THRESHOLD):
        return "Güven Düşük"
    
    curr_ankle_y = np.mean([curr_keypoints[15, 1], curr_keypoints[16, 1]])
    prev_ankly_y = np.mean([prev_keypoints[15,1], prev_keypoints[16,1]])
    diff_ankle = prev_ankly_y - curr_ankle_y

    curr_left_shoulder = curr_keypoints[5][:2]
    curr_right_shoulder = curr_keypoints[6][:2]
    shoulder_distance = np.linalg.norm(curr_left_shoulder - curr_right_shoulder) + 1e-6
    normalize_diff_ankly = diff_ankle / shoulder_distance

    def calculate_angle(p1, p2):
        delta_y = p2[1] - p1[1]
        delta_x = p2[0] - p1[0]
        return np.degrees(np.arctan2(delta_y, delta_x))
    
    curr_angle = calculate_angle(curr_left_shoulder, curr_right_shoulder)
    prev_left_shoulder = prev_keypoints[5][:2]
    prev_right_shoulder = prev_keypoints[6][:2]
    prev_angle = calculate_angle(prev_left_shoulder, prev_right_shoulder)
    diff_angle = curr_angle - prev_angle

    jump = normalize_diff_ankly > JUMP_THRESHOLD
    rotate = abs(diff_angle) > ROTATION_THRESHOLD

    if jump and rotate:
        return "Jump + Rotate"
    elif jump:
        return 'Jump'
    elif rotate:
        return "Rotate"
    else:
        return "Stationary"
    
print("Video")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    
    results = model(frame, show=False)
    print(f"Frame {frame_count} - raw results: {results}")

    if len(results) > 0 and hasattr(results[0], 'keypoints'):

        kp_all = results[0].keypoints.data.cpu().numpy()
        print(f"Frame {frame_count} - detected keypoints shape: {kp_all.shape}")

        if kp_all.shape[0] < 1 or kp_all.shape[1] < 17:
            print(f"Frame {frame_count} - keypoints {kp_all.shape}")
            movement_state = 'Hareketsiz'
        else:
            keypoints = kp_all[0]
            print(f"Frame {frame_count} - Ankle Y: {keypoints[15, 1]:.2f}, {keypoints[16, 1]:.2f}")
            print(f"Frame {frame_count} - Shoulder X: {keypoints[5, 0]:.2f}, {keypoints[6, 0]:.2f}")
            current_state = analyze_movement(keypoints, prev_keypoints)

            previous_states.append(current_state)

            if len(previous_states) > stability_frames:
                previous_states.pop(0)

            if len(previous_states) == stability_frames:

                from collections import Counter
                state_counter = Counter(previous_states)
                most_common_state = state_counter.most_common(1)[0][0]

                if state_counter[most_common_state] >= stability_frames * 0.6:
                    movement_state = most_common_state
                else:

                    movement_state = current_state
            else:
                movement_state = current_state

            prev_keypoints = keypoints
    else:

        if not previous_states or all(state == "Stationary" for state in previous_states[-3:]):
            movement_state = "Stationary"               

    print(f"Frame {frame_count} - Movement State: {movement_state}")

    annotated_frame = results[0].plot() if len(results) > 0 and hasattr(results[0], 'plot') else frame.copy()

    text = f"Movement: {movement_state}"

    font_scale = 1

    font = cv2.FONT_HERSHEY_COMPLEX

    thickness = 2

    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    x = width - text_width - 20

    y = 30 + text_height

    cv2.putText(annotated_frame, text, (x, y), font, font_scale, (0, 255, 0), thickness)

    writter.write(annotated_frame)

cap.release()
writter.release()

