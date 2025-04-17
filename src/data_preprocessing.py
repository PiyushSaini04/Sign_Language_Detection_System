"""
Data Preprocessing Module for Sign Language Detection ML Project

This module handles the preprocessing of collected hand landmark data,
including normalization, augmentation, and preparation for model training.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Define constants for landmark counts
POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
TOTAL_LANDMARKS_PER_SAMPLE = POSE_LANDMARKS * 3 + HAND_LANDMARKS * 3 + HAND_LANDMARKS * 3 # x,y,z for pose, left, right

class GestureDataProcessor:
    """
    Handles preprocessing of hand gesture landmark data.
    
    Attributes:
        data_dir (str): Directory containing raw gesture data files
        processed_dir (str): Directory to save processed data
        random_seed (int): Random seed for reproducibility
        label_encoder (LabelEncoder): Encoder for gesture labels
    """
    
    def __init__(self, data_dir=r'C:\Users\piyus\Downloads\Ai_project\Sign_Language_Detection_System-main\data\raw', processed_dir=r'C:\Users\piyus\Downloads\Ai_project\Sign_Language_Detection_System-main\data\processed', random_seed=42):
        """
        Initialize the GestureDataProcessor with specified parameters.
        
        Args:
            data_dir (str): Directory containing raw gesture data files
            processed_dir (str): Directory to save processed data
            random_seed (int): Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.random_seed = random_seed
        self.label_encoder = LabelEncoder()
        
        # Create processed directory if it doesn't exist
        os.makedirs(processed_dir, exist_ok=True)
    
    def load_gesture_data(self, file_pattern='*.json'):
        """
        Load all gesture data files from the data directory (new format).
        
        Args:
            file_pattern (str): Pattern to match gesture data files
            
        Returns:
            dict: Dictionary with gesture names as keys and lists of landmark sample dicts 
                  (containing 'pose', 'left_hand', 'right_hand') as values
        """
        gesture_data = {}
        
        # Find all gesture files
        data_files = glob.glob(os.path.join(self.data_dir, file_pattern))
        
        if not data_files:
            raise ValueError(f"No gesture data files found in {self.data_dir}")
        
        print(f"Found {len(data_files)} gesture data files.")
        
        # Load each file
        for file_path in tqdm(data_files, desc="Loading gesture data"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                gesture_name = data['gesture_name']
                landmarks_samples = data['landmarks_data'] # Changed key
                
                if gesture_name not in gesture_data:
                    gesture_data[gesture_name] = []
                
                # Basic validation: Check if samples have the expected keys
                valid_samples = []
                for sample in landmarks_samples:
                    if isinstance(sample, dict) and all(k in sample for k in ['pose', 'left_hand', 'right_hand']):
                        valid_samples.append(sample)
                    else:
                        print(f"Warning: Skipping invalid sample structure in {file_path}")
                        
                gesture_data[gesture_name].extend(valid_samples)

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from file: {file_path}")
            except KeyError as e:
                print(f"Warning: Missing key {e} in file: {file_path}")
        
        print("Loaded gesture data:")
        for gesture, samples in gesture_data.items():
            print(f"  - {gesture}: {len(samples)} samples")
        
        return gesture_data
    
    def normalize_landmarks(self, sample_data):
        """
        Normalize pose and hand landmarks relative to the pose.
        Uses shoulder center as origin and shoulder width as scale.
        
        Args:
            sample_data (dict): Dict containing 'pose', 'left_hand', 'right_hand' landmark lists.
            
        Returns:
            dict: Normalized landmark data in the same structure.
        """
        pose_landmarks = sample_data['pose']
        left_hand_landmarks = sample_data['left_hand']
        right_hand_landmarks = sample_data['right_hand']

        normalized_sample = {'pose': None, 'left_hand': None, 'right_hand': None}

        # --- Normalize Pose --- 
        if pose_landmarks and len(pose_landmarks) == POSE_LANDMARKS:
            pose_points = np.array([[p.get('x', 0), p.get('y', 0), p.get('z', 0)] for p in pose_landmarks])

            # Define reference points (ensure these indices are correct for mp.solutions.pose)
            left_shoulder = pose_points[11] 
            right_shoulder = pose_points[12]
            
            # Calculate center and scale reference
            shoulder_center = (left_shoulder + right_shoulder) / 2
            shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

            # Avoid division by zero if shoulders are coincident
            if shoulder_width > 1e-6: 
                # Normalize pose points
                centered_pose = pose_points - shoulder_center
                normalized_pose_points = centered_pose / shoulder_width
            else:
                normalized_pose_points = pose_points - shoulder_center # Only center if no scale

            # Convert back to list of dictionaries for pose
            normalized_pose_list = []
            for i, (px, py, pz) in enumerate(normalized_pose_points):
                 normalized_pose_list.append({
                     'x': float(px), 'y': float(py), 'z': float(pz),
                     'visibility': pose_landmarks[i].get('visibility', 0.0) # Preserve visibility
                 })
            normalized_sample['pose'] = normalized_pose_list
            
            # --- Normalize Hands (relative to the same pose origin/scale) --- 
            if shoulder_width > 1e-6: # Only normalize hands if pose normalization was successful
                # Normalize Left Hand
                if left_hand_landmarks and len(left_hand_landmarks) == HAND_LANDMARKS:
                    left_hand_points = np.array([[p.get('x', 0), p.get('y', 0), p.get('z', 0)] for p in left_hand_landmarks])
                    centered_left = left_hand_points - shoulder_center
                    normalized_left_points = centered_left / shoulder_width
                    # Convert back
                    normalized_sample['left_hand'] = [
                        {'x': float(px), 'y': float(py), 'z': float(pz)} 
                        for px, py, pz in normalized_left_points
                    ]
                
                # Normalize Right Hand
                if right_hand_landmarks and len(right_hand_landmarks) == HAND_LANDMARKS:
                    right_hand_points = np.array([[p.get('x', 0), p.get('y', 0), p.get('z', 0)] for p in right_hand_landmarks])
                    centered_right = right_hand_points - shoulder_center
                    normalized_right_points = centered_right / shoulder_width
                    # Convert back
                    normalized_sample['right_hand'] = [
                        {'x': float(px), 'y': float(py), 'z': float(pz)} 
                        for px, py, pz in normalized_right_points
                    ]
            else: # If pose normalization failed, don't normalize hands either (or handle differently)
                 if left_hand_landmarks: normalized_sample['left_hand'] = left_hand_landmarks
                 if right_hand_landmarks: normalized_sample['right_hand'] = right_hand_landmarks

        else: # Handle case where pose landmarks are missing or incomplete
            # Keep hands as they are if pose is missing
            normalized_sample['pose'] = pose_landmarks # Keep original or None
            normalized_sample['left_hand'] = left_hand_landmarks
            normalized_sample['right_hand'] = right_hand_landmarks
            print("Warning: Pose landmarks missing or incomplete, skipping normalization for this sample.")

        return normalized_sample
    
    def flatten_landmarks(self, normalized_sample):
        """
        Flatten normalized pose, left hand, and right hand landmarks into a 1D array.
        Handles missing hand data by filling with zeros.
        
        Args:
            normalized_sample (dict): Dict with normalized 'pose', 'left_hand', 'right_hand' lists.
            
        Returns:
            numpy.ndarray: Flattened array of landmark coordinates (pose + left + right).
                       Returns None if essential data (pose) is missing.
        """
        pose_landmarks = normalized_sample.get('pose')
        left_hand_landmarks = normalized_sample.get('left_hand')
        right_hand_landmarks = normalized_sample.get('right_hand')

        # Essential: require pose landmarks for a valid feature vector
        if not pose_landmarks or len(pose_landmarks) != POSE_LANDMARKS:
            print("Warning: Cannot flatten sample due to missing/incomplete pose landmarks.")
            return None 
        
        flattened = []
        
        # Add pose landmarks (x, y, z)
        for lm in pose_landmarks:
            flattened.extend([lm.get('x', 0), lm.get('y', 0), lm.get('z', 0)])
            # Optional: include visibility? flattened.extend([lm.get('visibility', 0)])
        
        # Add left hand landmarks (x, y, z) or zeros if missing
        if left_hand_landmarks and len(left_hand_landmarks) == HAND_LANDMARKS:
            for lm in left_hand_landmarks:
                flattened.extend([lm.get('x', 0), lm.get('y', 0), lm.get('z', 0)])
        else:
            flattened.extend([0.0] * (HAND_LANDMARKS * 3))
            
        # Add right hand landmarks (x, y, z) or zeros if missing
        if right_hand_landmarks and len(right_hand_landmarks) == HAND_LANDMARKS:
            for lm in right_hand_landmarks:
                flattened.extend([lm.get('x', 0), lm.get('y', 0), lm.get('z', 0)])
        else:
            flattened.extend([0.0] * (HAND_LANDMARKS * 3))
            
        # Verify expected length
        if len(flattened) != TOTAL_LANDMARKS_PER_SAMPLE:
             print(f"Warning: Flattened sample has unexpected length {len(flattened)}, expected {TOTAL_LANDMARKS_PER_SAMPLE}.")
             # Handle error appropriately, maybe return None or pad/truncate
             # For now, let's pad with zeros if too short, truncate if too long
             if len(flattened) < TOTAL_LANDMARKS_PER_SAMPLE:
                 flattened.extend([0.0] * (TOTAL_LANDMARKS_PER_SAMPLE - len(flattened)))
             else:
                 flattened = flattened[:TOTAL_LANDMARKS_PER_SAMPLE]

        return np.array(flattened)
    
    def augment_landmarks(self, normalized_sample, num_augmentations=5):
        """
        Generate augmented versions of normalized landmarks.
        Applies the same random transformation (rotation, translation, scale)
        to pose, left hand, and right hand landmarks consistently.
        
        Args:
            normalized_sample (dict): Dict with normalized landmark lists.
            num_augmentations (int): Number of augmented versions to generate.
            
        Returns:
            list: List of augmented landmark sample dictionaries.
        """
        augmented_sets = []
        pose_landmarks = normalized_sample.get('pose')
        left_hand_landmarks = normalized_sample.get('left_hand')
        right_hand_landmarks = normalized_sample.get('right_hand')

        # Need pose for augmentation reference
        if not pose_landmarks:
            return [] 

        for _ in range(num_augmentations):
            augmented_sample = {'pose': None, 'left_hand': None, 'right_hand': None}

            # Generate random transformation parameters
            theta_z = np.random.uniform(-0.15, 0.15) # Rotation around Z
            theta_y = np.random.uniform(-0.1, 0.1)   # Rotation around Y
            theta_x = np.random.uniform(-0.1, 0.1)   # Rotation around X
            translation = np.random.uniform(-0.05, 0.05, size=3)
            scale = np.random.uniform(0.95, 1.05)

            # --- Create Rotation Matrix (combine rotations) --- 
            rot_z = np.array([
                [np.cos(theta_z), -np.sin(theta_z), 0],
                [np.sin(theta_z), np.cos(theta_z), 0],
                [0, 0, 1]
            ])
            rot_y = np.array([
                [np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)]
            ])
            rot_x = np.array([
                [1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)]
            ])
            rotation_matrix = rot_z @ rot_y @ rot_x # Combine rotations

            # --- Function to apply transformation ---
            def apply_transform(landmarks_list):
                if not landmarks_list: return None
                points = np.array([[lm.get('x', 0), lm.get('y', 0), lm.get('z', 0)] for lm in landmarks_list])
                transformed_points = np.dot(points, rotation_matrix)
                translated_points = transformed_points + translation
                scaled_points = translated_points * scale
                # Convert back to list of dicts
                transformed_list = [
                    {'x': float(px), 'y': float(py), 'z': float(pz)} 
                    # Add visibility back for pose if needed
                    # 'visibility': landmarks_list[i].get('visibility', 0.0) if 'visibility' in landmarks_list[0] else None
                    for i, (px, py, pz) in enumerate(scaled_points)
                ]
                 # Add visibility back specifically for pose
                if landmarks_list and 'visibility' in landmarks_list[0]:
                    for i in range(len(transformed_list)):
                         transformed_list[i]['visibility'] = landmarks_list[i].get('visibility', 0.0)
                return transformed_list

            # Apply transformation to each part
            augmented_sample['pose'] = apply_transform(pose_landmarks)
            augmented_sample['left_hand'] = apply_transform(left_hand_landmarks)
            augmented_sample['right_hand'] = apply_transform(right_hand_landmarks)
            
            # Only add if pose augmentation was successful
            if augmented_sample['pose']:
                augmented_sets.append(augmented_sample)
        
        return augmented_sets
    
    def prepare_dataset(self, augment=True, test_size=0.2, val_size=0.1):
        """
        Prepare the full dataset for training (updated for new format).
        """
        # Load raw gesture data (new format)
        gesture_data = self.load_gesture_data()
        
        all_features = []
        all_labels = []
        class_names = sorted(list(gesture_data.keys())) # Sort for consistency
        
        self.label_encoder.fit(class_names)
        
        print("Processing, normalizing, and flattening landmarks...")
        for gesture_name, sample_list in tqdm(gesture_data.items(), desc="Processing gestures"):
            label = gesture_name
            for sample_data in sample_list:
                # Normalize landmarks
                normalized_sample = self.normalize_landmarks(sample_data)
                
                # Flatten landmarks into feature vector
                features = self.flatten_landmarks(normalized_sample)
                
                # Add original sample if valid
                if features is not None:
                    all_features.append(features)
                    all_labels.append(label)
                
                    # Generate augmentations if needed
                    if augment:
                        augmented_samples = self.augment_landmarks(normalized_sample)
                        for aug_sample in augmented_samples:
                            aug_features = self.flatten_landmarks(aug_sample)
                            if aug_features is not None:
                                all_features.append(aug_features)
                                all_labels.append(label)
        
        if not all_features:
             raise ValueError("No valid features could be extracted from the data. Check data quality and processing steps.")

        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Encode labels
        y_encoded = self.label_encoder.transform(y)
        
        # Calculate val_ratio early if needed for minimum sample check
        val_ratio = 0.0
        if val_size > 0 and (1.0 - test_size) > 0:
            val_ratio = val_size / (1.0 - test_size)

        # Perform train-test split
        # Ensure there's enough data for splitting
        if len(np.unique(y_encoded)) < 2:
             print("Warning: Only one class present after processing. Cannot stratify split.")
             stratify_opt = None
        else:
             # Check if any class has fewer samples than required for splits
             min_samples_per_class = np.min(np.bincount(y_encoded))
             # Estimate minimum needed for stratified splits (crude estimate)
             # Now val_ratio is defined before being used here
             min_needed = max(2, int(1 / (test_size * val_ratio)) if val_size > 0 and val_ratio > 0 else int(1/test_size) )
             if min_samples_per_class < min_needed:
                  print(f"Warning: Smallest class has only {min_samples_per_class} samples. Stratified split might fail or be unreliable.")
                  stratify_opt = None # Fallback to non-stratified
             else:
                  stratify_opt = y_encoded
        
        try:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=self.random_seed, stratify=stratify_opt
            )
        except ValueError as e:
             print(f"Error during train-test split (potentially due to class imbalance): {e}. Trying without stratification.")
             X_train_val, X_test, y_train_val, y_test = train_test_split(
                 X, y_encoded, test_size=test_size, random_state=self.random_seed # No stratification
             )

        
        # Split training data into train and validation
        if val_size > 0 and len(X_train_val) > 1: # Ensure there's data to split for validation
            # Check stratification for train/val split
            if stratify_opt is not None:
                min_samples_per_class_train_val = np.min(np.bincount(y_train_val))
                min_needed_val = max(2, int(1/val_ratio))
                if min_samples_per_class_train_val < min_needed_val:
                     print(f"Warning: Smallest class in train/val set has {min_samples_per_class_train_val} samples. Stratified validation split might fail.")
                     stratify_opt_val = None
                else:
                     stratify_opt_val = y_train_val
            else:
                 stratify_opt_val = None
            
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val, y_train_val, test_size=val_ratio, random_state=self.random_seed, stratify=stratify_opt_val
                )
            except ValueError as e:
                 print(f"Error during train-validation split (potentially due to class imbalance): {e}. Trying without stratification.")
                 # Adjust val_ratio if necessary to avoid creating empty set
                 if int(len(X_train_val) * val_ratio) < 1:
                      val_ratio = 1.0 / len(X_train_val)
                 X_train, X_val, y_train, y_val = train_test_split(
                     X_train_val, y_train_val, test_size=val_ratio, random_state=self.random_seed # No stratification
                 )
        else: # No validation set needed or not enough data
             X_train, y_train = X_train_val, y_train_val
             X_val, y_val = np.array([]), np.array([]) # Empty validation set

        
        print(f"Dataset preparation complete:")
        print(f"  - Training samples: {X_train.shape[0]}")
        if X_val.shape[0] > 0:
            print(f"  - Validation samples: {X_val.shape[0]}")
        print(f"  - Test samples: {X_test.shape[0]}")
        print(f"  - Feature dimension: {X_train.shape[1]}")
        print(f"  - Number of classes: {len(class_names)}")
        
        # Save processed data
        self.save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, class_names)
        
        return X_train, y_train, X_val, y_val, X_test, y_test, class_names

    def save_processed_data(self, X_train, y_train, X_val, y_val, X_test, y_test, class_names):
        """
        Save the processed data splits and metadata.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = os.path.join(self.processed_dir, f"processed_data_{timestamp}")
        
        np.savez_compressed(
            f"{filename_base}.npz", 
            X_train=X_train, y_train=y_train, 
            X_val=X_val, y_val=y_val, 
            X_test=X_test, y_test=y_test
        )
        
        metadata = {
            'timestamp': timestamp,
            'num_train': len(X_train),
            'num_val': len(X_val),
            'num_test': len(X_test),
            'feature_dim': X_train.shape[1] if X_train.shape[0] > 0 else (X_test.shape[1] if X_test.shape[0] > 0 else None),
            'num_classes': len(class_names),
            'class_names': class_names,
            'label_encoding': {label: int(code) for label, code in zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_))},
            'normalization_details': 'Normalized relative to shoulder center and width.',
            'augmentation_used': 'Geometric (rotation, translation, scale)', # Update if different
            'random_seed': self.random_seed
        }
        
        metadata_path = f"{filename_base}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Processed data saved to {filename_base}.npz")
        print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    # Example usage
    processor = GestureDataProcessor(data_dir=r'C:\Users\piyus\Downloads\Ai_project\Sign_Language_Detection_System-main\data\raw', processed_dir=r'C:\Users\piyus\Downloads\Ai_project\Sign_Language_Detection_System-main\data\processed')
    
    try:
        # Prepare dataset (assuming raw data exists)
        X_train, y_train, X_val, y_val, X_test, y_test, class_names = processor.prepare_dataset(
            augment=True, 
            test_size=0.2, 
            val_size=0.1
        )
        
        print("\nDataset successfully prepared.")
        print(f"Training features shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        
        # You would typically pass this data to the model training script next
        
    except ValueError as e:
        print(f"\nError during dataset preparation: {e}")
    except FileNotFoundError:
        print("\nError: Raw data directory or files not found. Run data collection first.") 