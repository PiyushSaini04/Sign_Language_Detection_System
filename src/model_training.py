"""
Model Training Module for Sign Language Detection ML Project

This module handles the training of the machine learning model for
gesture recognition using preprocessed hand landmark data.
"""

import os
import json
import time
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class GestureModelTrainer:
    """
    Handles the training of ML models for gesture recognition.
    
    Attributes:
        model_dir (str): Directory to save trained models
        processed_data_dir (str): Directory containing processed data files (*.npz, *_metadata.json)
        model (tf.keras.Model): The TensorFlow model
        history (tf.keras.callbacks.History): Training history
        class_names (list): List of gesture class names
        input_shape (tuple): Shape of input features
        random_seed (int): Random seed for reproducibility
    """
    
    def __init__(self, model_dir=r'C:\Users\piyus\Downloads\Ai_project - Copy\Sign_Language_Detection_System-main\models', processed_data_dir=r'C:\Users\piyus\Downloads\Ai_project\Sign_Language_Detection_System-main\data\processed', random_seed=42):
        """
        Initialize the GestureModelTrainer with specified parameters.
        
        Args:
            model_dir (str): Directory to save trained models
            processed_data_dir (str): Directory containing processed data files (*.npz, *_metadata.json)
            random_seed (int): Random seed for reproducibility
        """
        self.model_dir = model_dir
        self.processed_data_dir = processed_data_dir
        self.model = None
        self.history = None
        self.class_names = []
        self.input_shape = None
        self.random_seed = random_seed
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
    
    def load_processed_data(self, data_path=None):
        """
        Load the latest processed data and metadata, or a specific file.
        
        Args:
            data_path (str, optional): Path to specific .npz file. If None, loads the latest.
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
        Raises:
            FileNotFoundError: If no processed data or metadata is found.
        """
        if data_path:
            npz_file = data_path
            metadata_file = data_path.replace('.npz', '_metadata.json')
        else:
            # Find the latest processed data based on timestamp in filename
            search_pattern_npz = os.path.join(self.processed_data_dir, "processed_data_*.npz")
            list_of_npz_files = glob.glob(search_pattern_npz)
            if not list_of_npz_files:
                raise FileNotFoundError(f"No processed .npz files found in {self.processed_data_dir}")
            latest_npz_file = max(list_of_npz_files, key=os.path.getctime)
            npz_file = latest_npz_file
            metadata_file = latest_npz_file.replace('.npz', '_metadata.json')
        
        if not os.path.exists(npz_file):
            raise FileNotFoundError(f"Processed data file not found: {npz_file}")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        print(f"Loading processed data from: {npz_file}")
        print(f"Loading metadata from: {metadata_file}")
        
        # Load data
        data = np.load(npz_file)
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.class_names = metadata['class_names']
        self.input_shape = (metadata['feature_dim'],)
        num_classes = metadata['num_classes']
        
        print(f"Data loaded successfully. Input shape: {self.input_shape}, Classes: {len(self.class_names)}")
        
        # Basic validation
        if X_train.shape[1] != self.input_shape[0] or X_test.shape[1] != self.input_shape[0] or \
               (X_val.size > 0 and X_val.shape[1] != self.input_shape[0]):
             raise ValueError("Mismatch between metadata feature dimension and loaded data shape.")
        if len(self.class_names) != num_classes:
             raise ValueError("Mismatch between metadata class count and loaded class names.")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def build_model(self, num_classes, hidden_units=[128, 64]):
        """
        Build a sequential neural network model.
        Input shape is now determined by loaded data via self.input_shape.
        
        Args:
            num_classes (int): Number of output classes (gestures).
            hidden_units (list): List of integers specifying units in hidden dense layers.
            
        Returns:
            tf.keras.Model: Compiled Keras model.
        """
        if not self.input_shape:
            raise ValueError("Input shape not determined. Load data or a model first.")
        
        print(f"Building model with input shape: {self.input_shape} and {num_classes} classes.")
        
        model = models.Sequential(name="GestureRecognitionModel")
        model.add(layers.Input(shape=self.input_shape, name="input_landmarks"))
        model.add(layers.BatchNormalization())
        
        # Add hidden layers
        for units in hidden_units:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(0.3))
        
        # Output layer
        model.add(layers.Dense(num_classes, activation='softmax', name="output_probabilities"))
        
        # Compile the model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        self.model.summary()
        return model
    
    def train(self, X_train, y_train, X_val, y_val, class_names=None, epochs=50, batch_size=32, patience=10, save_best_only=True, model_type='dense'):
        """
        Train the gesture recognition model.
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels (encoded)
            X_val (numpy.ndarray): Validation features
            y_val (numpy.ndarray): Validation labels (encoded)
            class_names (list, optional): List of class names matching the encoded labels
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            patience (int): Early stopping patience
            save_best_only (bool): Whether to save only the best model
            model_type (str): Type of model to build (dense or lstm)
            
        Returns:
            tf.keras.callbacks.History: Training history
        """
        # Set class names if provided
        if class_names:
            self.class_names = class_names
            print(f"Using provided class names: {', '.join(self.class_names)}")
        
        if self.model is None:
            if not self.class_names:
                 raise ValueError("Cannot build model without class information. Load data first.")
            print("Model not built yet. Building default model.")
            self.build_model(num_classes=len(self.class_names))
        
        if X_train.size == 0 or y_train.size == 0:
             raise ValueError("Training data is empty.")
        
        print(f"Starting training for {epochs} epochs...")
        
        # Callbacks
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_checkpoint_path = os.path.join(self.model_dir, f"gesture_model_{timestamp}_best.h5")
        
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val.size > 0 else 'loss',
                patience=patience,
                verbose=1,
                restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                filepath=model_checkpoint_path,
                monitor='val_accuracy' if X_val.size > 0 else 'accuracy',
                save_best_only=save_best_only,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val.size > 0 else 'loss',
                factor=0.2,
                patience=5,
                verbose=1,
                min_lr=1e-6
            )
        ]
        
        # Use validation data if available
        validation_data = (X_val, y_val) if X_val.size > 0 and y_val.size > 0 else None
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks_list,
            verbose=1
        )
        
        print("Training complete.")
        
        # Save the final model (potentially the restored best one)
        final_model_path = os.path.join(self.model_dir, f"gesture_model_{timestamp}_final.h5")
        self.save_model(final_model_path)
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model on the test set.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        if X_test.size == 0 or y_test.size == 0:
             print("Warning: Test data is empty. Skipping evaluation.")
             return None, None
        
        print("\nEvaluating model on test set...")
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Generate predictions for detailed metrics
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        # Ensure class_names are available and match the prediction indices
        target_names = self.class_names if self.class_names else [str(i) for i in range(len(np.unique(y_test)))]
        # Adjust target_names length if necessary, though ideally class_names is loaded correctly
        num_unique_labels = len(np.unique(np.concatenate((y_test, y_pred))))
        if len(target_names) < num_unique_labels:
             print("Warning: Number of class names doesn't match unique labels in test/pred. Using numeric labels.")
             target_names = [str(i) for i in range(num_unique_labels)]
        elif len(target_names) > num_unique_labels:
             # This case is less likely but handle defensively
             target_names = target_names[:num_unique_labels] 
        
        # Filter labels present in y_test for classification report
        unique_test_labels = np.unique(y_test)
        filtered_target_names = [name for i, name in enumerate(target_names) if i in unique_test_labels]
        if not filtered_target_names:
             filtered_target_names = [str(i) for i in unique_test_labels]
        
        # Use labels parameter to handle cases where some classes might not appear in y_test
        report_labels = np.unique(np.concatenate((y_test, y_pred)))
        report_target_names = [name for i, name in enumerate(target_names) if i in report_labels]
        if not report_target_names:
             report_target_names = [str(i) for i in report_labels]
        elif len(report_target_names) < len(report_labels):
             # Fallback if name mapping is problematic
             report_target_names = [str(i) for i in report_labels]
        
        try:
             print(classification_report(y_test, y_pred, target_names=report_target_names, labels=report_labels, zero_division=0))
        except ValueError as e:
             print(f"Could not generate classification report with names: {e}. Printing with numeric labels.")
             print(classification_report(y_test, y_pred, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=report_labels)
        self.plot_confusion_matrix(cm, report_target_names)
        
        return loss, accuracy
    
    def plot_confusion_matrix(self, cm, target_names):
        """
        Plot confusion matrix for model evaluation.
        
        Args:
            cm (numpy.ndarray): Confusion matrix
            target_names (list): List of class names
        """
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names,
                   yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save the plot
        cm_path = os.path.join(self.model_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        
        print(f"Confusion matrix saved to {cm_path}")
    
    def plot_training_history(self):
        """
        Plot training history (accuracy and loss curves).
        """
        if not self.history:
            print("No training history available.")
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        history_path = os.path.join(self.model_dir, 'training_history.png')
        plt.savefig(history_path)
        plt.close()
        
        print(f"Training history plot saved to {history_path}")
    
    def save_model(self, filepath=None):
        """
        Save the trained model and necessary metadata (class names, input shape).
        Uses the HDF5 format (.h5).
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        if filepath is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filepath = os.path.join(self.model_dir, f"gesture_model_{timestamp}.h5")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model using HDF5 format
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
        # Save metadata separately (class names, input shape)
        metadata = {
            'class_names': self.class_names,
            'input_shape': self.input_shape,
            'model_format': '.h5'
        }
        metadata_path = filepath.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Model metadata saved to {metadata_path}")
        
        # Also save a backup of class names in root metadata dir for easier access
        try:
            root_metadata_path = os.path.join(self.model_dir, "class_names_metadata.json")
            with open(root_metadata_path, 'w') as f:
                json.dump({'class_names': self.class_names}, f, indent=2)
            print(f"Backup class names saved to {root_metadata_path}")
        except Exception as e:
            print(f"Warning: Could not save backup class names: {e}")
    
    def load_model(self, filepath=None):
        """
        Load a trained model and its metadata.
        If filepath is None, tries to load the latest .h5 model in the model directory.
        """
        import os
        import glob
        
        if filepath is None:
            # Find the latest .h5 model file
            list_of_h5_files = glob.glob(os.path.join(self.model_dir, "*.h5"))
            if not list_of_h5_files:
                print(f"No .h5 model files found in {self.model_dir}. Cannot load latest.")
                return None
            latest_file = max(list_of_h5_files, key=os.path.getctime)
            filepath = latest_file
        
        if not filepath.endswith(".h5") or not os.path.exists(filepath):
             # Compatibility check for older H5 format if needed
             if filepath.endswith(".h5") and os.path.exists(filepath):
                  print(f"Loading model from H5 format: {filepath}")
             elif not os.path.exists(filepath):
                  print(f"Model file not found: {filepath}")
                  return None
             else:
                  print(f"Model file expected to be .h5 format: {filepath}")
                  # Attempt loading anyway, TF might handle it
                  pass 
        
        metadata_path = filepath.replace('.h5', '_metadata.json')
        if not os.path.exists(metadata_path):
            print(f"Warning: Metadata file not found: {metadata_path}. Will try alternative sources for class names.")
            self.class_names = [] # Reset class names if metadata missing
            self.input_shape = None
            
            # Try to load class names from backup metadata file
            backup_metadata_path = os.path.join(os.path.dirname(filepath), "class_names_metadata.json")
            if os.path.exists(backup_metadata_path):
                try:
                    with open(backup_metadata_path, 'r') as f:
                        backup_metadata = json.load(f)
                        if 'class_names' in backup_metadata and backup_metadata['class_names']:
                            self.class_names = backup_metadata['class_names']
                            print(f"Loaded class names from backup metadata: {', '.join(self.class_names)}")
                except Exception as e:
                    print(f"Error loading backup metadata: {e}")
                    
            # If still no class names, try loading from processed data
            if not self.class_names:
                try:
                    processed_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(filepath))), "data", "processed")
                    metadata_files = glob.glob(os.path.join(processed_data_dir, "*_metadata.json"))
                    if metadata_files:
                        latest_metadata = max(metadata_files, key=os.path.getctime)
                        with open(latest_metadata, 'r') as f:
                            metadata = json.load(f)
                            if 'class_names' in metadata:
                                self.class_names = metadata['class_names']
                                print(f"Loaded class names from processed data: {', '.join(self.class_names)}")
                except Exception as e:
                    print(f"Error loading class names from processed data: {e}")
        else:
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.class_names = metadata['class_names']
                self.input_shape = tuple(metadata['input_shape']) # Metadata saves list, model needs tuple
                print(f"Loaded metadata: {len(self.class_names)} classes, input shape {self.input_shape}")
            except Exception as e:
                 print(f"Error loading metadata from {metadata_path}: {e}. Proceeding without metadata.")
                 self.class_names = []
                 self.input_shape = None
        
        try:
            self.model = models.load_model(filepath)
            print(f"Model loaded successfully from {filepath}")
            
            # Verify input shape if possible
            if self.input_shape and self.model.input_shape[1:] != self.input_shape:
                print(f"Warning: Loaded model input shape {self.model.input_shape[1:]} differs from metadata {self.input_shape}.")
                # Optionally raise error or try to adapt?
                # For now, update self.input_shape to match the loaded model
                self.input_shape = self.model.input_shape[1:]
                print(f"Updated input shape to match loaded model: {self.input_shape}")
            elif not self.input_shape:
                 # If metadata failed, get shape from model
                 self.input_shape = self.model.input_shape[1:]
                 print(f"Inferred input shape from loaded model: {self.input_shape}")

            # If class names were missing, we can't map output indices to names
            if not self.class_names:
                 print("Warning: Class names are missing. Prediction output will be indices.")
                 # Optionally try to infer number of classes from output layer
                 if hasattr(self.model.layers[-1], 'units'):
                      num_classes_model = self.model.layers[-1].units
                      self.class_names = [f"class_{i}" for i in range(num_classes_model)]
                      print(f"Inferred {num_classes_model} classes from model output layer.")
                
            self.model.summary() # Print summary of loaded model
            return self.model
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            self.model = None
            return None
    
    def predict(self, features):
        """
        Make a prediction for a single set of features (landmarks).
        Assumes features are already preprocessed (normalized, flattened).
        
        Args:
            features (numpy.ndarray): Flattened feature vector.
            
        Returns:
            tuple: (predicted_class_name, confidence)
        """
        if self.model is None:
            raise ValueError("No model loaded for prediction. Load a model first.")
        if self.input_shape and features.shape[0] != self.input_shape[0]:
             raise ValueError(f"Input feature length {features.shape[0]} does not match model expected input shape {self.input_shape[0]}")
        
        # Reshape for model input (expects batch dimension)
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Make prediction
        prediction_probs = self.model.predict(features)[0]
        
        # Get the top prediction index and confidence
        predicted_class_idx = np.argmax(prediction_probs)
        confidence = prediction_probs[predicted_class_idx]
        
        # Get class name using loaded class_names
        if self.class_names and predicted_class_idx < len(self.class_names):
            predicted_class_name = self.class_names[predicted_class_idx]
        else:
            # Fallback if class names aren't loaded correctly
            predicted_class_name = f"Class_{predicted_class_idx}"
            if not self.class_names:
                 print("Warning: Class names not loaded, returning index as name.")
        
        return predicted_class_name, float(confidence)

if __name__ == "__main__":
    # Example Usage
    trainer = GestureModelTrainer(model_dir=r'C:\Users\piyus\Downloads\Ai_project - Copy\Sign_Language_Detection_System-main\data\processed')
    
    try:
        # 1. Load the preprocessed data
        X_train, y_train, X_val, y_val, X_test, y_test = trainer.load_processed_data()
        
        # 2. Build the model (input shape is now known from loaded data)
        trainer.build_model(num_classes=len(trainer.class_names))
        
        # 3. Train the model
        history = trainer.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=64, patience=15)
        
        # 4. Plot training history
        if history:
            trainer.plot_training_history()
        
        # 5. Evaluate the model on the test set
        trainer.evaluate(X_test, y_test)
        
        # 6. (Optional) Load the best saved model and evaluate again
        print("\nLoading best saved model for final evaluation...")
        # Find the best model saved during training (now looking for .h5)
        list_of_model_files = glob.glob(os.path.join(trainer.model_dir, "*.h5"))
        if list_of_model_files: 
            # Assuming the naming convention includes '_best'
            best_models = [f for f in list_of_model_files if '_best' in os.path.basename(f)]
            if best_models:
                 latest_best_model = max(best_models, key=os.path.getctime)
                 print(f"Found best model: {latest_best_model}")
                 loaded_model = trainer.load_model(latest_best_model)
                 if loaded_model:
                      trainer.evaluate(X_test, y_test)
            else:
                 print("Could not find a model file with '_best' in the name (ending in .h5).")
        else:
             print("No .h5 model files found to load the best one.")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure you have run the data collection and preprocessing scripts first.")
    except ValueError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}") 