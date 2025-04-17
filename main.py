import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Open video capture
cap = cv2.VideoCapture(0)

# Variables for jumping jack detection
jumping = False  # Track state (open position)
count = 0  # Jumping jack counter

with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get necessary landmarks (normalized values: 0 to 1)
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
            left_foot = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y
            right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y

            # Calculate conditions for a jumping jack
            hands_up = left_wrist < left_shoulder and right_wrist < right_shoulder
            feet_apart = abs(left_foot - right_foot) > abs(left_hip - right_hip) * 1.5

            # Detect transition from 'Open' to 'Closed' position
            if hands_up and feet_apart:
                jumping = True  # In open position
            elif jumping and not hands_up and not feet_apart:
                jumping = False  # Transition to closed position
                count += 1  # Increase count when complete cycle detected

            # Display count on frame
            cv2.putText(frame, f"Jumping Jacks: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show frame
        cv2.imshow('Jumping Jack Counter', frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()