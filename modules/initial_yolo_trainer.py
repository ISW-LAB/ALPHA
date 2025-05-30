# modules/initial_yolo_trainer.py
"""
Initial YOLO Model Training Module
Module for training YOLO models with various data ratios
"""

import os
import random
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import numpy as np

class InitialYOLOTrainer:
    def __init__(self, dataset_root, images_dir, labels_dir, output_dir, 
                 model_type='yolov8n.pt', epochs=100, img_size=640, batch_size=16,
                 percentages=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                 train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, random_seed=13):
        """
        Initialize Initial YOLO Trainer
        
        Args:
            dataset_root (str): Dataset root directory
            images_dir (str): Original images directory
            labels_dir (str): Original labels directory
            output_dir (str): Model save directory
            model_type (str): Base YOLO model type
            epochs (int): Number of training epochs
            img_size (int): Image size
            batch_size (int): Batch size
            percentages (list): List of data percentages to train
            train_ratio (float): Training data ratio
            valid_ratio (float): Validation data ratio
            test_ratio (float): Test data ratio
            random_seed (int): Random seed
        """
        self.dataset_root = dataset_root
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.output_dir = output_dir
        self.model_type = model_type
        self.epochs = epochs
        self.img_size = img_size
        self.batch_size = batch_size
        self.percentages = percentages
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Directory setup
        self.split_dataset_root = os.path.join(self.dataset_root, 'dataset')
        self.train_dir = os.path.join(self.split_dataset_root, 'train')
        self.valid_dir = os.path.join(self.split_dataset_root, 'valid')
        self.test_dir = os.path.join(self.split_dataset_root, 'test')
        self.temp_dir = os.path.join(self.dataset_root, 'temp_train')
        
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        print("Creating necessary directories...")
        
        # Create dataset split directories
        for dir_path in [self.train_dir, self.valid_dir, self.test_dir]:
            os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
            os.makedirs(os.path.join(dir_path, 'labels'), exist_ok=True)
        
        # Create temporary training directory
        os.makedirs(os.path.join(self.temp_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'labels'), exist_ok=True)
        
        # Create model save directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("Directory setup completed")
    
    def split_dataset(self):
        """Split original images and labels into train/valid/test"""
        print("Starting dataset splitting...")
        
        # Get all image files
        image_files = [f for f in os.listdir(self.images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            raise ValueError(f"No images found: {self.images_dir}")
        
        print(f"Found {len(image_files)} image files")
        
        # Set seed for consistent random selection
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Randomly shuffle image files
        random.shuffle(image_files)
        
        # Calculate dataset split
        total_images = len(image_files)
        train_size = int(total_images * self.train_ratio)
        valid_size = int(total_images * self.valid_ratio)
        
        # Split dataset
        train_files = image_files[:train_size]
        valid_files = image_files[train_size:train_size+valid_size]
        test_files = image_files[train_size+valid_size:]
        
        print(f"Data split: Train={len(train_files)}, Valid={len(valid_files)}, Test={len(test_files)}")
        
        # Clear split directories (remove existing files)
        for dir_path in [self.train_dir, self.valid_dir, self.test_dir]:
            for subdir in ['images', 'labels']:
                subdir_path = os.path.join(dir_path, subdir)
                if os.path.exists(subdir_path):
                    for file in os.listdir(subdir_path):
                        os.remove(os.path.join(subdir_path, file))
        
        # Function to copy files
        def copy_files(file_list, source_images, source_labels, dest_dir, split_name):
            copied_count = 0
            for img_file in file_list:
                # Copy image file
                src_img = os.path.join(source_images, img_file)
                dst_img = os.path.join(dest_dir, 'images', img_file)
                
                # Check label filename (change extension to .txt)
                label_file = os.path.splitext(img_file)[0] + '.txt'
                src_label = os.path.join(source_labels, label_file)
                dst_label = os.path.join(dest_dir, 'labels', label_file)
                
                # Copy only if both image and corresponding label exist
                if os.path.exists(src_img) and os.path.exists(src_label):
                    shutil.copy(src_img, dst_img)
                    
                    # Copy label file (convert to single class)
                    with open(src_label, 'r') as original_label:
                        lines = original_label.readlines()
                    
                    with open(dst_label, 'w') as new_label:
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:  # Format: class_id x y w h
                                # Set class ID to 0 (single class)
                                parts[0] = '0'
                                new_label.write(' '.join(parts) + '\n')
                    
                    copied_count += 1
                else:
                    if not os.path.exists(src_img):
                        print(f"Warning: Image file not found - {src_img}")
                    if not os.path.exists(src_label):
                        print(f"Warning: Label file not found - {src_label}")
            
            print(f"{split_name} data copying completed: {copied_count} files")
            return copied_count
        
        # Execute file copying
        train_copied = copy_files(train_files, self.images_dir, self.labels_dir, self.train_dir, "Train")
        valid_copied = copy_files(valid_files, self.images_dir, self.labels_dir, self.valid_dir, "Valid")
        test_copied = copy_files(test_files, self.images_dir, self.labels_dir, self.test_dir, "Test")
        
        print(f"Dataset splitting completed:")
        print(f"  - Training data: {train_copied} files ({train_copied/total_images*100:.1f}%)")
        print(f"  - Validation data: {valid_copied} files ({valid_copied/total_images*100:.1f}%)")
        print(f"  - Test data: {test_copied} files ({test_copied/total_images*100:.1f}%)")
        
        if train_copied == 0:
            raise ValueError("No training data available. Please check image and label files.")
        
        return train_copied
    
    def create_subset(self, percentage):
        """Select a portion of training data based on given percentage (stratified approach)"""
        # Get all images from training directory
        train_images_dir = os.path.join(self.train_dir, 'images')
        image_files = [f for f in os.listdir(train_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            raise ValueError(f"No training images found: {train_images_dir}")
        
        total_images = len(image_files)
        subset_size = int(total_images * percentage / 100)
        
        print(f"Creating data subset: {percentage}% ({subset_size}/{total_images})")
        
        # Set seed for consistent ordering
        random.seed(self.random_seed)
        shuffled_images = random.sample(image_files, len(image_files))
        selected_images = shuffled_images[:subset_size]
        
        # Clear temporary directory
        temp_images_dir = os.path.join(self.temp_dir, 'images')
        temp_labels_dir = os.path.join(self.temp_dir, 'labels')
        
        for file in os.listdir(temp_images_dir):
            os.remove(os.path.join(temp_images_dir, file))
        for file in os.listdir(temp_labels_dir):
            os.remove(os.path.join(temp_labels_dir, file))
        
        # Copy selected images and corresponding labels
        copied_count = 0
        for image_file in selected_images:
            # Copy image
            src_img = os.path.join(self.train_dir, 'images', image_file)
            dst_img = os.path.join(temp_images_dir, image_file)
            
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
                
                # Copy label (same filename with .txt extension)
                label_file = os.path.splitext(image_file)[0] + '.txt'
                src_label = os.path.join(self.train_dir, 'labels', label_file)
                dst_label = os.path.join(temp_labels_dir, label_file)
                
                if os.path.exists(src_label):
                    shutil.copy(src_label, dst_label)
                    copied_count += 1
                else:
                    print(f"Warning: Label file not found - {src_label}")
            else:
                print(f"Warning: Image file not found - {src_img}")
        
        print(f"Subset creation completed: {copied_count} files copied")
        return copied_count, total_images
    
    def create_dataset_yaml(self):
        """Create dataset YAML file"""
        yaml_data = {
            'path': os.path.dirname(os.path.abspath(self.temp_dir)),
            'train': os.path.abspath(os.path.join(self.temp_dir, 'images')),
            'val': os.path.abspath(os.path.join(self.valid_dir, 'images')),
            'test': os.path.abspath(os.path.join(self.test_dir, 'images')),
            'nc': 1,
            'names': {0: 'object'}
        }
        
        yaml_path = os.path.join(self.output_dir, 'temp_data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
        
        print(f"Dataset YAML file created: {yaml_path}")
        return yaml_path
    
    def train_with_percentage(self, percentage):
        """Train model with specific percentage of data"""
        print(f"\n{'='*60}")
        print(f"=== Starting training with {percentage}% data ===")
        print(f"{'='*60}")
        
        # Create subset of training data
        subset_size, total_images = self.create_subset(percentage)
        
        if subset_size == 0:
            print(f"Warning: No training data available for {percentage}% data.")
            return None
        
        print(f"Selected {subset_size} out of {total_images} images ({percentage}%)")
        
        # Create dataset YAML file
        temp_yaml = self.create_dataset_yaml()
        
        # Initialize model
        print(f"Initializing YOLO model: {self.model_type}")
        model = YOLO(self.model_type)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Start model training
            print(f"Starting model training...")
            print(f"  - Number of images: {subset_size} ({percentage}%)")
            print(f"  - Epochs: {self.epochs}")
            print(f"  - Image size: {self.img_size}")
            print(f"  - Batch size: {self.batch_size}")
            
            results = model.train(
                data=temp_yaml,
                epochs=self.epochs,
                imgsz=self.img_size,
                batch=self.batch_size,
                name=f"yolov8_{percentage}pct_{timestamp}",
                patience=15,  # Early stopping setting
                save_period=10,  # Save every 10 epochs
                plots=True,  # Save training graphs
                verbose=True
            )
            
            # Copy trained best weights to output directory
            run_dir = Path(f"runs/detect/yolov8_{percentage}pct_{timestamp}")
            best_weights = run_dir / "weights" / "best.pt"
            
            if best_weights.exists():
                output_path = os.path.join(self.output_dir, f"yolov8_{percentage}pct.pt")
                shutil.copy(best_weights, output_path)
                print(f"✅ Best weights saved: {output_path}")
                
                # Also copy training logs
                results_dir = os.path.join(self.output_dir, f"training_results_{percentage}pct")
                if run_dir.exists():
                    shutil.copytree(run_dir, results_dir, dirs_exist_ok=True)
                    print(f"📊 Training results saved: {results_dir}")
                
                return output_path
            else:
                print(f"❌ Warning: Best weights not found at {best_weights}")
                return None
                
        except Exception as e:
            print(f"❌ Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_all_percentages(self):
        """Execute training for all percentages"""
        print("="*80)
        print("Starting Initial YOLO Model Training")
        print("="*80)
        
        # First, split the dataset
        print("Step 1: Splitting dataset into train/valid/test...")
        try:
            train_count = self.split_dataset()
        except Exception as e:
            print(f"❌ Dataset splitting failed: {str(e)}")
            return {}
        
        if train_count == 0:
            print("❌ Error: No training data available. Please check image and label directories.")
            return {}
        
        # Dictionary to store model paths
        trained_models = {}
        successful_count = 0
        failed_count = 0
        
        print(f"\nStep 2: Starting training with various data ratios")
        print(f"Training ratios: {self.percentages}")
        
        # Repeat training with various percentages
        for i, percentage in enumerate(self.percentages):
            print(f"\n🔄 Progress: {i+1}/{len(self.percentages)} - Training with {percentage}%")
            
            try:
                model_path = self.train_with_percentage(percentage)
                if model_path and os.path.exists(model_path):
                    trained_models[percentage] = model_path
                    successful_count += 1
                    print(f"✅ {percentage}% training completed: {model_path}")
                else:
                    failed_count += 1
                    print(f"❌ {percentage}% training failed")
            except Exception as e:
                failed_count += 1
                print(f"❌ Exception during {percentage}% training: {str(e)}")
        
        # Clean up temporary directory
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                print("🧹 Temporary directory cleanup completed")
        except Exception as e:
            print(f"⚠️ Temporary directory cleanup failed: {str(e)}")
        
        # Result summary
        print("\n" + "="*80)
        print("Initial YOLO Model Training Completed")
        print("="*80)
        print(f"📊 Training Results Summary:")
        print(f"  - Successful: {successful_count}")
        print(f"  - Failed: {failed_count}") 
        print(f"  - Total attempts: {len(self.percentages)}")
        print(f"  - Model save location: {self.output_dir}")
        
        if trained_models:
            print(f"\n✅ Successfully trained models:")
            for percentage, path in trained_models.items():
                print(f"  - {percentage}%: {path}")
        
        return trained_models

if __name__ == "__main__":
    # Test execution
    trainer = InitialYOLOTrainer(
        dataset_root='./dataset',
        images_dir='./dataset/images',
        labels_dir='./dataset/labels',
        output_dir='./models/initial_yolo',
        model_type='yolov8n.pt',
        epochs=50,  # Reduced value for testing
        percentages=[10, 50, 100]  # Reduced values for testing
    )
    
    results = trainer.train_all_percentages()
    print(f"Training completed! Results: {results}")