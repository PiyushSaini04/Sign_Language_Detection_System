"""
Data Collection Module for Sign Language Detection ML Project

This module handles the collection of hand gesture data using camera input
and MediaPipe for hand landmark detection.
"""

import os
import cv2
import time
import json
import numpy as np
import mediapipe as mp
from datetime import datetime
from tqdm import tqdm

# Import pose solution
mp_pose = mp.solutions.pose

class DataCollector:
    """
    Handles the collection of hand gesture data using computer vision.
    
    Attributes:
        output_dir (str): Directory to save collected data
        mp_hands (mediapipe.solutions.hands): MediaPipe Hands solution
        mp_pose (mediapipe.solutions.pose): MediaPipe Pose solution
        hands (mediapipe.solutions.hands.Hands): Hand detector instance
        pose (mediapipe.solutions.pose.Pose): Pose detector instance
        mp_drawing (mediapipe.solutions.drawing_utils): MediaPipe drawing utilities
        mp_drawing_styles (mediapipe.solutions.drawing_styles): MediaPipe drawing styles
        capture_delay (float): Delay between frame captures in seconds
        min_detection_confidence (float): Minimum confidence for hand detection
        min_tracking_confidence (float): Minimum confidence for hand tracking
        min_pose_detection_confidence (float): Minimum confidence for pose detection
        min_pose_tracking_confidence (float): Minimum confidence for pose tracking
    """
    
    def __init__(self, 
                 output_dir=r'C:\Users\piyus\Downloads\Ai_project\Sign_Language_Detection_System-main\data\raw',
                 capture_delay=0.1,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5,
                 min_pose_detection_confidence=0.5,
                 min_pose_tracking_confidence=0.5):
        """
        Initialize the DataCollector with specified parameters.
        
        Args:
            output_dir (str): Directory to save collected data
            capture_delay (float): Delay between frame captures in seconds
            min_detection_confidence (float): Minimum confidence for hand detection
            min_tracking_confidence (float): Minimum confidence for hand tracking
            min_pose_detection_confidence (float): Minimum confidence for pose detection
            min_pose_tracking_confidence (float): Minimum confidence for pose tracking
        """
        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=min_pose_detection_confidence,
            min_tracking_confidence=min_pose_tracking_confidence
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Capture parameters
        self.capture_delay = capture_delay
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.min_pose_detection_confidence = min_pose_detection_confidence
        self.min_pose_tracking_confidence = min_pose_tracking_confidence
    
    def collect_gesture_data(self, gesture_name, num_samples=100):
        """
        Collect hand gesture data from webcam.
        
        Args:
            gesture_name (str): Name of the gesture to collect
            num_samples (int): Number of samples to collect
            
        Returns:
            str: Path to the saved gesture data file
        """
        print(f"Preparing to collect {num_samples} samples for gesture: {gesture_name}")
        print("Press 'SPACE' to start collection, 'Q' to quit")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Error: Could not open webcam.")
        
        # Variables for collection process
        samples_collected = 0
        landmarks_data = []
        start_collection = False
        last_capture_time = 0
        
        # Configure camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        progress_bar = tqdm(total=num_samples, desc=f"Collecting {gesture_name}", leave=False)

        while samples_collected < num_samples:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Flip the frame horizontally for a more intuitive display
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Process the frame for hands and pose
            hand_results = self.hands.process(rgb_frame)
            pose_results = self.pose.process(rgb_frame)
            
            # Draw annotations
            rgb_frame.flags.writeable = True
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # Draw Pose landmarks
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    pose_results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # Draw Hand landmarks
            if hand_results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            # Display status information on the frame
            status_text = f"Gesture: {gesture_name} | "
            if start_collection:
                status_text += f"Collecting: {samples_collected}/{num_samples}"
            else:
                status_text += "Press SPACE to start"
            
            cv2.putText(
                frame, 
                status_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            
            # Display the frame
            cv2.imshow('Hand Gesture Data Collection', frame)
            
            # Capture landmarks if collection has started
            if start_collection and (time.time() - last_capture_time) > self.capture_delay:
                if pose_results.pose_landmarks and hand_results.multi_hand_landmarks:
                    
                    current_sample = {'pose': None, 'left_hand': None, 'right_hand': None}

                    # Extract Pose landmarks
                    pose_landmarks_list = []
                    for lm in pose_results.pose_landmarks.landmark:
                         pose_landmarks_list.append({
                            'x': lm.x, 'y': lm.y, 'z': lm.z,
                            'visibility': lm.visibility if hasattr(lm, 'visibility') else 1.0
                         })
                    current_sample['pose'] = pose_landmarks_list

                    # Extract Hand landmarks with handedness
                    for hand_landmarks, handedness_info in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                        hand_label = handedness_info.classification[0].label.lower()
                        
                        hand_landmarks_list = []
                        for lm in hand_landmarks.landmark:
                            hand_landmarks_list.append({
                                'x': lm.x, 'y': lm.y, 'z': lm.z,
                            })
                        
                        if hand_label == 'left':
                             current_sample['left_hand'] = hand_landmarks_list
                        elif hand_label == 'right':
                             current_sample['right_hand'] = hand_landmarks_list

                    # Only add if we have the required landmarks (adjust as needed)
                    landmarks_data.append(current_sample)
                    samples_collected += 1
                    last_capture_time = time.time()
                    progress_bar.update(1)

                    # Check if we've collected enough samples (moved inside if)
                    # if samples_collected >= num_samples:
                    #     break # Exit inner loop, the outer loop condition handles this
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nCollection aborted by user.")
                progress_bar.close()
                samples_collected = num_samples
                landmarks_data = []
                break 
            elif key == ord(' ') and not start_collection:  # Space key - start only once
                print("\nStarting data collection...")
                start_collection = True
                last_capture_time = time.time()
        
        progress_bar.close()

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Save collected data
        if landmarks_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{gesture_name}_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            gesture_data = {
                'gesture_name': gesture_name,
                'timestamp': timestamp,
                'num_samples': len(landmarks_data),
                'landmarks_data': landmarks_data
            }
            
            with open(filepath, 'w') as f:
                json.dump(gesture_data, f, indent=2)
            
            print(f"\nSaved {len(landmarks_data)} samples to {filepath}")
            return filepath
        else:
            print("\nNo valid data collected.")
            return None
    
    def collect_multiple_gestures(self, gestures_config):
        """
        Collect data for multiple gestures.
        
        Args:
            gestures_config (list): List of dictionaries with 'name' and 'samples' keys
            
        Returns:
            list: Paths to the saved gesture data files
        """
        saved_files = []
        
        for gesture in gestures_config:
            gesture_name = gesture['name']
            num_samples = gesture.get('samples', 100)
            
            print(f"\n{'='*40}")
            print(f"Collecting data for gesture: {gesture_name}")
            print(f"{'='*40}\n")
            
            filepath = self.collect_gesture_data(gesture_name, num_samples)
            if filepath:
                saved_files.append(filepath)
            
            # Small delay between gestures
            time.sleep(1)
        
        return saved_files

# Removed extract_landmarks_from_file as it needs complete rewrite for new format
# Will handle loading in the preprocessing script

if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    
    # Define gestures to collect
    gestures = [
        {'name': 'call', 'samples': 50},
        {'name': 'hello', 'samples': 50},
        {'name': 'I love you', 'samples': 50},
        {'name': 'no', 'samples': 50},
        {'name': 'Thank you', 'samples': 50},
        {'name': 'Think', 'samples': 50},
        {'name': 'up', 'samples': 50},
        {'name': 'yes', 'samples': 50},
        {'name': 'yours', 'samples': 50},
    ]
    
    # Collect data for all gestures
    saved_files = collector.collect_multiple_gestures(gestures)
    
    print("\nData collection completed!")
    print(f"Collected data for {len(saved_files)} gestures.") 