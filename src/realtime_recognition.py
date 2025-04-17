"""
Real-time Recognition Module for Sign Language Detection ML Project

This module handles real-time gesture recognition using a webcam,
allowing for interactive sign language detection.
"""

import os
import cv2
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

from data_preprocessing import GestureDataProcessor, POSE_LANDMARKS, HAND_LANDMARKS, TOTAL_LANDMARKS_PER_SAMPLE
from model_training import GestureModelTrainer

# Define constants for landmark indices (using MediaPipe standard indices)
LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12

class RealtimeGestureRecognizer:
    """
    Handles real-time gesture recognition using webcam input (updated for pose + 2 hands).
    
    Attributes:
        model_path (str): Path to the trained model (.keras file)
        min_detection_confidence (float): Minimum confidence for hand detection
        min_tracking_confidence (float): Minimum confidence for hand tracking
        min_pose_detection_confidence (float): Minimum confidence for pose detection
        min_pose_tracking_confidence (float): Minimum confidence for pose tracking
        recognition_threshold (float): Minimum confidence threshold for gesture recognition
        smoothing_window (int): Window size for temporal smoothing
        history_buffer (deque): Buffer for storing recent predictions
        sequence_buffer (list): Buffer for storing gesture sequences
        sequence_timeout (float): Timeout for sequence detection in seconds
        last_gesture_time (float): Timestamp of last recognized gesture
        trainer (GestureModelTrainer): Instance for loading model and getting class names
        processor (GestureDataProcessor): Instance for normalizing landmarks
        mp_hands (mediapipe.solutions.hands): MediaPipe Hands solution
        mp_pose (mediapipe.solutions.pose): MediaPipe Pose solution
        mp_drawing (mediapipe.solutions.drawing_utils): MediaPipe drawing utilities
        mp_drawing_styles (mediapipe.solutions.drawing_styles): MediaPipe drawing styles
        hands (mediapipe.solutions.hands.Hands): Hand detector instance
        pose (mediapipe.solutions.pose.Pose): Pose detector instance
    """
    
    def __init__(self, 
                 model_path=r"C:\Users\piyus\Downloads\Ai_project - Copy\Sign_Language_Detection_System-main\models\gesture_model_20250417-004252_best.h5", 
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5,
                 min_pose_detection_confidence=0.5,
                 min_pose_tracking_confidence=0.5,
                 recognition_threshold=0.9,
                 smoothing_window=10):
        """
        Initialize the RealtimeGestureRecognizer.
        
        Args:
            model_path (str): Path to the trained model (.keras file)
            min_detection_confidence (float): Minimum confidence for hand detection
            min_tracking_confidence (float): Minimum confidence for hand tracking
            min_pose_detection_confidence (float): Minimum confidence for pose detection
            min_pose_tracking_confidence (float): Minimum confidence for pose tracking
            recognition_threshold (float): Minimum confidence threshold for gesture recognition
            smoothing_window (int): Window size for temporal smoothing
        """
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=min_pose_detection_confidence,
            min_tracking_confidence=min_pose_tracking_confidence
        )
        
        # Default class names placeholder 
        self.class_names = []
        
        # First, try to load class names from processed data (most reliable source)
        try:
            import glob
            import json
            import os
            
            processed_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
            metadata_files = glob.glob(os.path.join(processed_data_dir, "*_metadata.json"))
            
            if metadata_files:
                latest_metadata = max(metadata_files, key=os.path.getctime)
                with open(latest_metadata, 'r') as f:
                    metadata = json.load(f)
                    if 'class_names' in metadata and metadata['class_names']:
                        self.class_names = metadata['class_names']
                        print(f"Using class names from processed data: {', '.join(self.class_names)}")
        except Exception as e:
            print(f"Failed to load class names from processed data: {e}")
        
        # Model setup
        self.trainer = GestureModelTrainer()
        if model_path:
            self.model = self.trainer.load_model(model_path)
        else:
            self.model = self.trainer.load_model()
        
        # If we didn't get class names from the processed data, try model metadata
        if not self.class_names and hasattr(self.trainer, 'class_names') and self.trainer.class_names:
            # Check if names look valid (not just class_0, class_1)
            valid_names = [name for name in self.trainer.class_names if not name.startswith('class_')]
            if len(valid_names) == len(self.trainer.class_names):
                self.class_names = self.trainer.class_names
                print(f"Using class names from model metadata: {', '.join(self.class_names)}")
        
        # If we still don't have class names, log an error
        if not self.class_names:
            print("WARNING: Could not load class names from either processed data or model metadata")
        
        # Recognition parameters
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.min_pose_detection_confidence = min_pose_detection_confidence
        self.min_pose_tracking_confidence = min_pose_tracking_confidence
        self.recognition_threshold = recognition_threshold
        
        # Smoothing and sequence detection
        self.smoothing_window = smoothing_window
        self.history_buffer = deque(maxlen=smoothing_window)
        self.sequence_buffer = []
        self.sequence_timeout = 2.0  # seconds
        self.last_gesture_time = 0
        
        # Processor for landmark normalization
        self.processor = GestureDataProcessor()
    
    def preprocess_landmarks_live(self, pose_results, hand_results):
        """
        Preprocess pose and hand landmarks captured live for model input.
        Uses the normalization and flattening logic from GestureDataProcessor.

        Args:
            pose_results: MediaPipe Pose results object.
            hand_results: MediaPipe Hands results object.

        Returns:
            numpy.ndarray: Flattened feature vector for model input, or None if insufficient data.
        """
        # Extract landmarks into the dictionary format expected by the processor
        sample_data = {'pose': None, 'left_hand': None, 'right_hand': None}

        # Extract Pose
        if pose_results and pose_results.pose_landmarks:
            pose_list = []
            for lm in pose_results.pose_landmarks.landmark:
                pose_list.append({'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility})
            if len(pose_list) == POSE_LANDMARKS:
                sample_data['pose'] = pose_list
            else:
                print(f"Warning: Detected pose has {len(pose_list)} landmarks, expected {POSE_LANDMARKS}")

        # Extract Hands
        if hand_results and hand_results.multi_hand_landmarks:
            for hand_landmarks, handedness_info in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                hand_label = handedness_info.classification[0].label.lower()
                hand_list = []
                for lm in hand_landmarks.landmark:
                    hand_list.append({'x': lm.x, 'y': lm.y, 'z': lm.z})
                
                if len(hand_list) == HAND_LANDMARKS:
                    if hand_label == 'left':
                        sample_data['left_hand'] = hand_list
                    elif hand_label == 'right':
                        sample_data['right_hand'] = hand_list
                else:
                    print(f"Warning: Detected {hand_label} hand has {len(hand_list)} landmarks, expected {HAND_LANDMARKS}")

        # Check if we have at least pose data (required for normalization/flattening)
        if not sample_data['pose']:
            # print("Debug: No pose detected, cannot generate features.") # Optional debug print
            return None

        # Normalize using the processor method
        normalized_sample = self.processor.normalize_landmarks(sample_data)
        
        # Flatten using the processor method
        features = self.processor.flatten_landmarks(normalized_sample)

        # Validate feature length before returning
        if features is not None and len(features) != TOTAL_LANDMARKS_PER_SAMPLE:
            print(f"Warning: Preprocessed features have unexpected length {len(features)}, expected {TOTAL_LANDMARKS_PER_SAMPLE}. Returning None.")
            return None
        # elif features is None:
        #      print("Debug: Flattening returned None.") # Optional debug print

        return features
    
    def get_smoothed_prediction(self):
        """
        Get smoothed prediction based on recent history.
        
        Returns:
            tuple: (gesture_name, confidence) or (None, 0) if no clear prediction
        """
        if not self.history_buffer:
            return None, 0.0
        
        # Count occurrences of each gesture
        gesture_counts = {}
        gesture_confidences = {}
        
        for gesture, confidence in self.history_buffer:
            if gesture not in gesture_counts:
                gesture_counts[gesture] = 0
                gesture_confidences[gesture] = 0.0
            
            gesture_counts[gesture] += 1
            gesture_confidences[gesture] += confidence
        
        # Find most frequent gesture
        max_count = 0
        max_gesture = None
        
        for gesture, count in gesture_counts.items():
            if count > max_count:
                max_count = count
                max_gesture = gesture
        
        # Check if dominant gesture appears in more than 60% of the window
        # Adjust threshold as needed based on performance
        smoothing_threshold = 0.6 
        if max_gesture is not None and max_count / len(self.history_buffer) >= smoothing_threshold:
            avg_confidence = gesture_confidences[max_gesture] / max_count
            return max_gesture, avg_confidence
        
        return None, 0.0
    
    def update_sequence(self, gesture, confidence):
        """
        Update gesture sequence with new detection.
        
        Args:
            gesture (str): Recognized gesture
            confidence (float): Recognition confidence
        """
        current_time = time.time()
        
        # Check if we should start a new sequence (timeout exceeded)
        if current_time - self.last_gesture_time > self.sequence_timeout:
            self.sequence_buffer = []
        
        # Update last gesture time
        self.last_gesture_time = current_time
        
        # Only add to sequence if it's different from the last one
        if not self.sequence_buffer or self.sequence_buffer[-1][0] != gesture:
            self.sequence_buffer.append((gesture, confidence))
            
            # Limit sequence length
            if len(self.sequence_buffer) > 10:
                self.sequence_buffer.pop(0)
    
    def get_sequence_text(self):
        """
        Get current gesture sequence as text.
        
        Returns:
            str: Space-separated gesture sequence
        """
        return ' '.join([item[0] for item in self.sequence_buffer])
    
    def predict(self, features):
        """
        Override trainer's predict method to ensure proper class name handling
        """
        if self.model is None:
            return None, 0.0
            
        # Reshape for model input (expects batch dimension)
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Make prediction
        prediction_probs = self.model.predict(features)[0]
        
        # Get the top prediction index and confidence
        predicted_class_idx = np.argmax(prediction_probs)
        confidence = prediction_probs[predicted_class_idx]
        
        # Get class name using our class mapping
        if predicted_class_idx < len(self.class_names):
            predicted_class_name = self.class_names[predicted_class_idx]
        else:
            # Fallback if index out of range
            predicted_class_name = f"Unknown_{predicted_class_idx}"
            print(f"Warning: Index {predicted_class_idx} out of range for class names array of length {len(self.class_names)}")
        
        return predicted_class_name, float(confidence)
    
    def run(self, camera_id=0, flip_image=True):
        """
        Run real-time gesture recognition with webcam feed (updated).
        
        Args:
            camera_id (int): Camera device ID
            flip_image (bool): Whether to flip the camera image horizontally
        """
        # Check if model and class names are loaded
        if self.model is None:
            raise ValueError("Model not loaded. Provide a valid model path or ensure one exists.")
        
        if not self.class_names:
            raise ValueError("Class names not available. Check model metadata or loading process.")
        
        print("Starting real-time gesture recognition (Pose + 2 Hands)...")
        print(f"Loaded {len(self.class_names)} gestures: {', '.join(self.class_names)}")
        print("Recognition settings: checks every 2 seconds, minimum accuracy 0.9")
        
        # Initialize webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera with ID {camera_id}")
        
        # Get camera frame dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create a larger window to accommodate webcam, text box, and buttons
        window_width = frame_width
        window_height = frame_height + 200  # Extra space for text box and buttons
        
        # Create the main window
        cv2.namedWindow('Sign Language Recognition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Sign Language Recognition', window_width, window_height)
        
        # Variables to store recognition results
        detected_text = ""
        
        # Variables for timed recognition (every 2 seconds)
        last_recognition_time = 0
        recognition_interval = 2.0  # Check for gestures every 2 seconds
        
        # Function to handle button clicks
        def handle_mouse_events(event, x, y, flags, param):
            nonlocal detected_text
            
            if event == cv2.EVENT_LBUTTONDOWN:
                # Clear All button
                if 50 <= x <= 220 and window_height - 80 <= y <= window_height - 20:
                    detected_text = ""
                    self.sequence_buffer = []
                    self.history_buffer.clear()
                
                # Save to Text File button
                elif 240 <= x <= 440 and window_height - 80 <= y <= window_height - 20:
                    if detected_text:
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        file_path = f"sign_language_text_{timestamp}.txt"
                        with open(file_path, "w") as f:
                            f.write(detected_text)
                        print(f"Text saved to {file_path}")
                
                # Quit button
                elif 460 <= x <= 600 and window_height - 80 <= y <= window_height - 20:
                    cap.release()
                    cv2.destroyAllWindows()
                    self.hands.close()
                    self.pose.close()
                    exit(0)
        
        # Set mouse callback
        cv2.setMouseCallback('Sign Language Recognition', handle_mouse_events)
        
        # Main loop
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Error reading from camera")
                break
            
            # Flip the image horizontally
            if flip_image:
                image = cv2.flip(image, 1)
            
            # --- MediaPipe Processing --- 
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False # Improve performance
            hand_results = self.hands.process(image_rgb)
            pose_results = self.pose.process(image_rgb)
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Create annotated image for display
            annotated_image = image_bgr.copy()
            
            # Create a canvas for the full UI (webcam + text box + buttons)
            ui_canvas = np.ones((window_height, window_width, 3), dtype=np.uint8) * 240  # Light gray background
            
            # Check if hands are in the frame
            hands_in_frame = hand_results and hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) > 0
            
            # --- Landmark Preprocessing and Prediction ---
            current_time = time.time()
            time_since_last_check = current_time - last_recognition_time
            
            # Default status messages
            prediction_text = "No sign detection" if not hands_in_frame else ""
            confidence_text = ""
            
            # Process hands and detection in real-time
            if hands_in_frame:
                # Draw Hand landmarks and orange bounding box
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Create orange bounding box around the hand
                    x_min, y_min = frame_width, frame_height
                    x_max, y_max = 0, 0
                    
                    # Find the bounding box coordinates
                    for landmark in hand_landmarks.landmark:
                        px = int(landmark.x * frame_width)
                        py = int(landmark.y * frame_height)
                        x_min = min(x_min, px)
                        y_min = min(y_min, py)
                        x_max = max(x_max, px)
                        y_max = max(y_max, py)
                    
                    # Add padding to the bounding box
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(frame_width, x_max + padding)
                    y_max = min(frame_height, y_max + padding)
                    
                    # Draw orange bounding box
                    cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 165, 255), 2)
                
                # Process detection in every frame (continuous real-time display)
                features = self.preprocess_landmarks_live(pose_results, hand_results)
                
                if features is not None:
                    # Make prediction for display on screen
                    gesture, confidence = self.predict(features)
                    
                    # Always show the detected gesture and confidence on screen
                    prediction_text = f"Detected: {gesture}"
                    confidence_text = f"Conf: {confidence:.2f}"
                    
                    # Add to history buffer for smoothing
                    self.history_buffer.append((gesture, confidence))
                    
                    # Get smoothed prediction
                    smooth_gesture, smooth_confidence = self.get_smoothed_prediction()
                    
                    # Only update the textbox every 2 seconds with high confidence detections
                    if time_since_last_check >= recognition_interval and smooth_confidence >= self.recognition_threshold:
                        # Update sequence
                        self.update_sequence(smooth_gesture, smooth_confidence)
                        
                        # Update detected text with new gestures
                        if len(self.sequence_buffer) == 1:  # If this is a new sequence
                            if detected_text and not detected_text.endswith("\n"):
                                detected_text += "\n"
                            detected_text += smooth_gesture
                        elif len(self.sequence_buffer) > 1 and self.sequence_buffer[-1] != self.sequence_buffer[-2]:
                            detected_text += " " + smooth_gesture
                        
                        # Update the last recognition time
                        last_recognition_time = current_time
            
            # Draw Pose landmarks if present
            if pose_results and pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            # Place the annotated image on the UI canvas
            ui_canvas[0:frame_height, 0:frame_width] = annotated_image
            
            # Add text overlay for current detection
            cv2.putText(ui_canvas, prediction_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            if confidence_text:
                cv2.putText(ui_canvas, confidence_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Draw text box area (labeled "H" in the image)
            text_box_y = frame_height + 10
            text_box_height = 80
            cv2.rectangle(ui_canvas, (10, text_box_y), (window_width - 10, text_box_y + text_box_height), (200, 200, 200), -1)  # Gray background
            cv2.rectangle(ui_canvas, (10, text_box_y), (window_width - 10, text_box_y + text_box_height), (0, 0, 0), 1)  # Black border
            
            # Add 'H' label to the left of the text box
            cv2.putText(ui_canvas, "", (20, text_box_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
            
            # Display the detected text in the text box
            # Split text into lines to fit in the box
            if detected_text:
                text_x, text_y = 50, text_box_y + 20
                line_height = 20
                
                # Simple text wrapping
                words = detected_text.split()
                lines = []
                current_line = ""
                
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    if cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] < window_width - 100:
                        current_line = test_line
                    else:
                        lines.append(current_line)
                        current_line = word
                
                if current_line:
                    lines.append(current_line)
                
                # Display text (showing last 2 lines if there are more)
                display_lines = lines[-2:] if len(lines) > 2 else lines
                for i, line in enumerate(display_lines):
                    cv2.putText(ui_canvas, line, (text_x, text_y + i * line_height), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            
            # Draw buttons
            button_y = window_height - 80
            button_height = 60
            
            # Clear All button
            cv2.rectangle(ui_canvas, (50, button_y), (220, button_y + button_height), (200, 230, 230), -1)  # Light beige/yellow in BGR
            cv2.rectangle(ui_canvas, (50, button_y), (220, button_y + button_height), (0, 0, 0), 1)  # Black border
            cv2.putText(ui_canvas, "Clear All", (90, button_y + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            
            # Save to a Text File button
            cv2.rectangle(ui_canvas, (240, button_y), (440, button_y + button_height), (0, 150, 0), -1)  # Green in BGR
            cv2.rectangle(ui_canvas, (240, button_y), (440, button_y + button_height), (0, 0, 0), 1)  # Black border
            cv2.putText(ui_canvas, "Save", (260, button_y + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            
            # Quit button
            cv2.rectangle(ui_canvas, (460, button_y), (600, button_y + button_height), (0, 0, 255), -1)  # Red in BGR
            cv2.rectangle(ui_canvas, (460, button_y), (600, button_y + button_height), (0, 0, 0), 1)  # Black border
            cv2.putText(ui_canvas, "Quit", (505, button_y + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)  # White text
            
            # Display the UI
            cv2.imshow('Sign Language Recognition', ui_canvas)
            
            # Handle key presses
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Clear the sequence buffer and detected text
                self.sequence_buffer = []
                self.history_buffer.clear()
                detected_text = ""
                print("Sequence and history cleared")
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close() # Release MediaPipe resources
        self.pose.close()  # Release MediaPipe resources

if __name__ == "__main__":
    # Find the latest model automatically or specify a path
    # model_path = "../models/gesture_model_YYYYMMDD-HHMMSS_best.keras" # Example path
    model_path = None # Let trainer load latest

    try:
        recognizer = RealtimeGestureRecognizer(
            model_path=r"C:\Users\piyus\Downloads\Ai_project - Copy\Sign_Language_Detection_System-main\models\gesture_model_20250417-004252_best.h5", 
            min_detection_confidence=0.6, # Adjust thresholds as needed
            min_tracking_confidence=0.5,
            min_pose_detection_confidence=0.5,
            min_pose_tracking_confidence=0.5,
            recognition_threshold=0.9, # Higher threshold for better accuracy
            smoothing_window=5 # Smaller window might be more responsive
        )
        recognizer.run(camera_id=0, flip_image=True)
    except ValueError as e:
        print(f"Error initializing or running recognizer: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure model file and metadata exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 