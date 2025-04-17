# Sign Language Detection System using Machine Learning

A Python-based machine learning system for detecting and recognizing sign language gestures in real-time using computer vision and deep learning.

## Overview

This project implements a complete pipeline for sign language detection and recognition:

1. **Data Collection**: Capture custom sign language data using webcam, MediaPipe Hands, and MediaPipe Pose (for two hands and body/arm context).
2. **Data Preprocessing**: Normalize and augment pose and hand landmark data relative to the body pose for training.
3. **Model Training**: Train neural network models to recognize gestures based on combined pose and hand landmarks.
4. **Real-time Recognition**: Perform real-time gesture recognition and sequence detection using pose and hand tracking.

The system allows users to train their own custom gestures, making it flexible for different sign languages and user preferences.

## Project Structure

```
sign_language_ml/
├── data/                  # Data storage
│   ├── raw/               # Raw gesture data (JSON files - pose + 2 hands)
│   └── processed/         # Processed datasets (NPZ + metadata JSON)
├── models/                # Trained models (.keras + metadata JSON)
├── notebooks/             # Jupyter notebooks (for exploration and visualization)
├── src/                   # Source code
│   ├── data_collection.py # Gesture data collection module (Pose + 2 Hands)
│   ├── data_preprocessing.py # Data preprocessing module (Pose + 2 Hands)
│   ├── model_training.py  # Model training module (accepts Pose + 2 Hands features)
│   ├── realtime_recognition.py # Real-time recognition module (Pose + 2 Hands)
│   └── main.py            # Main CLI entry point (may need updates for new data/models)
└── requirements.txt       # Python dependencies
README.md              # This file
```

## Detailed Module Explanation

### `data_collection.py`

This module handles collecting sign language data using your webcam and MediaPipe's **Pose** and **Hands** tracking solutions.

**Key Components**:

- **`DataCollector` Class**: Main class for collecting gesture data
  - **`__init__()`**: Initializes MediaPipe Hands (for 2 hands) and Pose, sets up capture parameters, and configures output directory.
  - **`collect_gesture_data()`**: Captures video frames, extracts landmarks for pose (33 landmarks), left hand (21 landmarks), and right hand (21 landmarks), including handedness.
  - **`collect_multiple_gestures()`**: Allows collecting multiple gestures in sequence.

**How It Works**:

1. Initializes camera and MediaPipe pose and hand detectors.
2. Displays live webcam feed with pose and hand landmarks overlaid.
3. When capture starts, records landmark positions (x, y, z coordinates) for pose, left hand, and right hand separately for each frame.
4. Saves the combined landmark data in JSON format, structured to differentiate pose, left hand, and right hand data (handling missing hands). Visibility data from pose is also stored.

### `data_preprocessing.py`

This module handles the preprocessing of raw pose and hand landmark data into a format suitable for training machine learning models.

**Key Components**:

- **`GestureDataProcessor` Class**: Processes raw gesture data.
  - **`load_gesture_data()`**: Loads gesture data from the new JSON format (containing pose, left_hand, right_hand).
  - **`normalize_landmarks()`**: Normalizes pose and hand landmarks relative to a stable pose reference (e.g., shoulder center and width) to make them invariant to scale and translation.
  - **`flatten_landmarks()`**: Converts normalized 3D landmarks for pose, left hand, and right hand into a single flat feature vector, handling missing hands with zero-padding.
  - **`augment_landmarks()`**: Generates augmented data by applying consistent geometric transformations (rotations, translations, scaling) to pose and hand landmarks.
  - **`prepare_dataset()`**: Full pipeline that loads, normalizes, augments, flattens, and splits the dataset.
  - **`save_processed_data()`**: Saves processed feature vectors (X) and labels (y) in `.npz` format, along with a `_metadata.json` file containing class names, feature dimensions, etc.

**How It Works**:

1. Loads raw landmark data (pose, left hand, right hand) from JSON files.
2. Normalizes landmarks based on the pose (e.g., relative to shoulder center/width) for invariance.
3. Augments data by applying transformations consistently across pose and hands.
4. Flattens the normalized landmarks into a single large feature vector per sample.
5. Splits data into training, validation, and test sets.
6. Saves processed data (`.npz`) and metadata (`.json`).

### `model_training.py`

This module handles training and evaluating machine learning models using the combined pose and hand features.

**Key Components**:

- **`GestureModelTrainer` Class**: Trains and evaluates models.
  - **`load_processed_data()`**: Loads the processed `.npz` data file and its corresponding `_metadata.json` file.
  - **`build_model()`**: Creates a neural network (e.g., dense sequential) with an input layer sized appropriately for the combined pose and hand features (dimension obtained from metadata).
  - **`train()`**: Trains the model with the loaded processed data.
  - **`save_model()`**: Saves the trained model in `.keras` format and a corresponding `_metadata.json` file (containing class names, input shape).
  - **`load_model()`**: Loads a previously trained `.keras` model and its metadata.
  - **`evaluate()`**: Evaluates model performance on the test set.
  - **`plot_confusion_matrix()`**, **`plot_training_history()`**: Visualization helpers.
  - **`predict()`**: Makes predictions on a single flattened feature vector.

**How It Works**:

1. Loads processed feature data and metadata.
2. Builds a neural network architecture suitable for the large input feature dimension.
3. Trains the model on the combined pose/hand landmark data.
4. Uses callbacks like EarlyStopping and ModelCheckpoint.
5. Evaluates performance using standard metrics.
6. Saves the trained model (`.keras`) and necessary metadata (`.json`).

### `realtime_recognition.py`

This module performs real-time sign language recognition using pose and two-hand tracking with a trained model.

**Key Components**:

- **`RealtimeGestureRecognizer` Class**: Performs real-time recognition.
  - **`__init__()`**: Initializes MediaPipe Pose and Hands (2 hands), loads the trained model (`.keras`) and metadata using `GestureModelTrainer`.
  - **`preprocess_landmarks_live()`**: Captures live pose and hand landmarks, combines them, normalizes them using `GestureDataProcessor` logic (relative to pose), and flattens them into the feature vector expected by the model. Handles missing hands.
  - **`get_smoothed_prediction()`**: Applies temporal smoothing.
  - **`update_sequence()`**, **`get_sequence_text()`**: Manage gesture sequences.
  - **`run()`**: Main loop capturing video, processing landmarks, predicting, and displaying results (including drawing pose and hand overlays).

**How It Works**:

1. Captures webcam feed and detects pose and hand landmarks using MediaPipe.
2. Extracts, normalizes, and flattens landmarks in real-time to match the model's input format.
3. Makes predictions using the loaded trained model.
4. Applies temporal smoothing and sequence logic.
5. Displays the webcam feed with pose and hand overlays, recognized gestures, and confidence scores.

### `main.py`

This module provides a command-line interface. **Note:** The CLI commands might need adjustments to reflect the changes in data format, model saving/loading (.keras + metadata), and potentially new arguments for pose confidence thresholds.

## Installation

1. Clone the repository.
2. Create and activate a Python virtual environment (e.g., using `venv` or `conda`).
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

**(Note: Command-line arguments in `main.py` might need updating)**

### Data Collection (Pose + 2 Hands)

```bash
# Example (verify arguments in main.py)
python src/main.py collect --gestures hello help thank_you --samples 100
```
Follow on-screen instructions. Press SPACE to start/stop recording for each gesture.

### Data Preprocessing (Pose + 2 Hands)

```bash
# Example (verify arguments in main.py)
python src/main.py preprocess --augment
```
This loads raw JSON data, processes pose and hand landmarks, and saves `.npz` and `_metadata.json` files in `data/processed/`.

### Model Training (Pose + 2 Hands)

```bash
# Example (verify arguments in main.py)
python src/main.py train --epochs 100 --batch-size 64
```
This loads the latest processed data, trains the model, and saves the best/final model (`.keras`) and metadata (`.json`) in `models/`.

### Model Evaluation

```bash
# Example (verify arguments in main.py)
python src/main.py evaluate --model-path models/your_trained_model.keras
```
Evaluates the specified trained model using the test set from the processed data.

### Real-time Recognition (Pose + 2 Hands)

```bash
# Example (verify arguments in main.py)
python src/main.py recognize --model-path models/your_trained_model.keras --threshold 0.7
```
Runs real-time recognition using the specified model.

## Technical Details

### Landmark Detection

- **MediaPipe Pose**: Detects 33 landmarks for the upper body, providing context and arm positioning.
- **MediaPipe Hands**: Detects 21 landmarks per hand (up to two hands), capturing finger and palm configuration. Handedness (Left/Right) is also detected.

### Data Normalization

- Normalization is now relative to the pose (e.g., using shoulder center as origin and shoulder width for scale). This makes the system robust to user position and distance from the camera.

### Model Architecture

- The input layer now accepts a significantly larger feature vector combining pose (33 * 3), left hand (21 * 3), and right hand (21 * 3) coordinates. Example: 99 + 63 + 63 = 225 features (adjust if visibility/other features are added).
- Hidden layer sizes might need adjustment to handle the increased input complexity.

### Palm vs. Back of Hand

- While not explicitly labeled, the 3D coordinates (`z` value) of the landmarks, especially wrist and knuckle points relative to finger tips, implicitly encode the hand's orientation (palm vs. back). The model can learn this distinction if the training data includes varied examples.

## Performance Considerations

- Processing pose *and* two hands increases computational load compared to just one hand. Performance (FPS) might decrease on less powerful hardware.
- The increased feature vector size requires a potentially larger model, which could slightly impact training time and prediction latency.

## Customization

- **Custom Gestures**: Collect and train your own gesture set
- **Model Tuning**: Adjust hyperparameters in `model_training.py`
- **Recognition Sensitivity**: Modify confidence threshold and smoothing window

## License

This project is open source and available under the MIT License.

## Acknowledgments

- MediaPipe for the hand tracking technology
- TensorFlow for the deep learning framework #   S i g n _ L a n g u a g e _ D e t e c t i o n _ S y s t e m  
 