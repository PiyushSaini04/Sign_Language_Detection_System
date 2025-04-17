"""
Main Entry Point for Sign Language Detection ML Project

This script provides a command-line interface to run different components
of the sign language detection system.
"""

import os
import argparse
import sys

def main():
    """
    Main function that parses command line arguments and runs the appropriate module.
    """
    parser = argparse.ArgumentParser(
        description='Sign Language Detection ML System',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Data collection command
    collect_parser = subparsers.add_parser('collect', help='Collect gesture data')
    collect_parser.add_argument('--gestures', nargs='+', required=True,
                               help='Names of gestures to collect (e.g. hello thank_you)')
    collect_parser.add_argument('--samples', type=int, default=50,
                               help='Number of samples to collect per gesture (default: 50)')
    collect_parser.add_argument('--output', type=str, default='../data/raw',
                               help='Output directory for collected data (default: ../data/raw)')
    
    # Data preprocessing command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess collected data')
    preprocess_parser.add_argument('--augment', action='store_true',
                                 help='Perform data augmentation')
    preprocess_parser.add_argument('--input', type=str, default='../data/raw',
                                 help='Input directory containing raw data (default: ../data/raw)')
    preprocess_parser.add_argument('--output', type=str, default='../data/processed',
                                 help='Output directory for processed data (default: ../data/processed)')
    
    # Model training command
    train_parser = subparsers.add_parser('train', help='Train gesture recognition model')
    train_parser.add_argument('--model-type', type=str, choices=['dense', 'lstm'], default='dense',
                            help='Type of model to train (default: dense)')
    train_parser.add_argument('--epochs', type=int, default=50,
                            help='Number of training epochs (default: 50)')
    train_parser.add_argument('--batch-size', type=int, default=32,
                            help='Training batch size (default: 32)')
    train_parser.add_argument('--data', type=str, default='../data/processed',
                            help='Directory containing processed data (default: ../data/processed)')
    train_parser.add_argument('--output', type=str, default='../models',
                            help='Output directory for trained model (default: ../models)')
    
    # Model evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model', type=str, default=None,
                           help='Path to trained model (default: latest in ../models)')
    eval_parser.add_argument('--data', type=str, default='../data/processed',
                           help='Directory containing processed data (default: ../data/processed)')
    
    # Real-time recognition command
    recognition_parser = subparsers.add_parser('recognize', help='Run real-time recognition')
    recognition_parser.add_argument('--model', type=str, default=None,
                                  help='Path to trained model (default: latest in ../models)')
    recognition_parser.add_argument('--camera', type=int, default=0,
                                  help='Camera device ID (default: 0)')
    recognition_parser.add_argument('--threshold', type=float, default=0.7,
                                  help='Recognition confidence threshold (default: 0.7)')
    recognition_parser.add_argument('--no-flip', action='store_true',
                                  help='Disable horizontal flipping of camera image')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if command was provided
    if not args.command:
        parser.print_help()
        return
    
    # Run appropriate module based on command
    if args.command == 'collect':
        from data_collection import DataCollector
        
        # Create list of gesture configs
        gestures = [{'name': name, 'samples': args.samples} for name in args.gestures]
        
        # Create data collector and collect data
        collector = DataCollector(output_dir=args.output)
        saved_files = collector.collect_multiple_gestures(gestures)
        
        print(f"\nData collection completed!")
        print(f"Collected data for {len(saved_files)} gestures:")
        for gesture in args.gestures:
            print(f"  - {gesture}")
        print(f"Data saved to {args.output}")
    
    elif args.command == 'preprocess':
        from data_preprocessing import GestureDataProcessor
        
        # Create data processor and process data
        processor = GestureDataProcessor(
            data_dir=args.input,
            processed_dir=args.output
        )
        
        X_train, y_train, X_val, y_val, X_test, y_test, class_names = processor.prepare_dataset(
            augment=args.augment
        )
        
        print("\nData preprocessing completed!")
        print(f"Processed data saved to {args.output}")
    
    elif args.command == 'train':
        from model_training import GestureModelTrainer
        from data_preprocessing import GestureDataProcessor
        
        # Load processed data
        processor = GestureDataProcessor(processed_dir=args.data)
        
        try:
            # Load processed data
            X_train, y_train, X_val, y_val, X_test, y_test, class_names = processor.load_processed_data()
            
            # Create model trainer and train model
            trainer = GestureModelTrainer(model_dir=args.output)
            
            history = trainer.train(
                X_train, y_train, X_val, y_val, class_names,
                epochs=args.epochs,
                batch_size=args.batch_size,
                model_type=args.model_type
            )
            
            # Evaluate the model
            evaluation = trainer.evaluate(X_test, y_test)
            
            print("\nModel training completed!")
            print(f"Model saved to {args.output}")
        
        except FileNotFoundError:
            print("Error: Processed data not found. Run preprocessing first.")
            return
    
    elif args.command == 'evaluate':
        from model_training import GestureModelTrainer
        from data_preprocessing import GestureDataProcessor
        
        # Load processed data
        processor = GestureDataProcessor(processed_dir=args.data)
        
        try:
            # Load processed data
            X_train, y_train, X_val, y_val, X_test, y_test, class_names = processor.load_processed_data()
            
            # Create model trainer and load model
            trainer = GestureModelTrainer()
            trainer.load_model(args.model)
            
            # Evaluate the model
            evaluation = trainer.evaluate(X_test, y_test)
            
            print("\nModel evaluation completed!")
        
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
    
    elif args.command == 'recognize':
        from realtime_recognition import RealtimeGestureRecognizer
        
        # Create recognizer and run real-time recognition
        model_path = args.model
        if model_path is None:
            # Find the best model in the models directory if not specified
            import glob
            import os
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            h5_files = glob.glob(os.path.join(models_dir, "*.h5"))
            if h5_files:
                model_path = max(h5_files, key=os.path.getctime)
                print(f"Using latest model: {model_path}")
            else:
                print("No model files found. Please train a model first or specify a model path.")
                return
        
        recognizer = RealtimeGestureRecognizer(
            model_path=model_path,
            recognition_threshold=args.threshold
        )
        
        try:
            recognizer.run(
                camera_id=args.camera,
                flip_image=not args.no_flip
            )
        except Exception as e:
            print(f"Error during recognition: {e}")
            return

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
        sys.exit(0) 