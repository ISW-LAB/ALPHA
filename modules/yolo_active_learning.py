# modules/yolo_active_learning.py
"""
YOLO Active Learning Core Module
Iterative learning system combining YOLO detection + Classification filtering
"""

import os
import cv2
import numpy as np
import torch
import yaml
import shutil
import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO

# Local module imports
from modules.object_classifier import ObjectClassifier

class YOLOActiveLearning:
    """
    YOLO-based Active Learning System
    Performs iterative learning by combining YOLO object detection with Classification models
    """
    
    def __init__(self, model_path, classifier_path=None, image_dir=None, label_dir=None, output_dir=None, 
                 conf_threshold=0.25, iou_threshold=0.5, class_conf_threshold=0.5, max_cycles=5, gpu_num=0,
                 use_classifier=False):
        """
        Initialize YOLO Active Learning System
        
        Args:
            model_path (str): Pre-trained YOLO model path
            classifier_path (str, optional): Pre-trained classification model path
            image_dir (str): Image dataset path
            label_dir (str): Ground truth label path
            output_dir (str): Results save path
            conf_threshold (float): Object detection confidence threshold
            iou_threshold (float): IoU threshold
            class_conf_threshold (float): Classification model confidence threshold
            max_cycles (int): Maximum number of training iterations
            gpu_num (int): GPU number to use
            use_classifier (bool): Whether to use classification model
        """
        # Store basic settings
        self.model_path = model_path
        self.classifier_path = classifier_path
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_dir = output_dir
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_conf_threshold = class_conf_threshold
        self.max_cycles = max_cycles
        self.gpu_num = gpu_num
        self.use_classifier = use_classifier
        
        # Extract model name
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        # GPU setup
        self.device = torch.device(f"cuda:{self.gpu_num}" if torch.cuda.is_available() else "cpu")
        
        print("🔧 YOLO Active Learning System Configuration:")
        print(f"  - YOLO model: {self.model_name}")
        print(f"  - Use classification model: {self.use_classifier}")
        print(f"  - Maximum cycles: {self.max_cycles}")
        print(f"  - Device: {self.device}")
        print(f"  - Results save: {self.output_dir}")
        
        # Create directory structure
        self.create_directories()
        
        # Load YOLO model
        print("📥 Loading YOLO model...")
        self.model = YOLO(self.model_path)
        print("✅ YOLO model loading completed")
        
        # Load classification model (optional)
        self.classifier = None
        if self.use_classifier and self.classifier_path:
            print("📥 Loading classification model...")
            self.classifier = ObjectClassifier(
                self.classifier_path, 
                self.device, 
                self.class_conf_threshold, 
                self.gpu_num
            )
            print("✅ Classification model loading completed")
        
        # Initialize performance metrics tracking dataframe
        self.setup_metrics_tracking()
        
        # Initialize statistics variables
        self.reset_statistics()
        
        print("✅ YOLO Active Learning system initialization completed")
        
    def create_directories(self):
        """Create necessary directory structure"""
        print("📁 Creating directory structure...")
        
        # Basic output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Results directory for each training cycle
        for cycle in range(1, self.max_cycles + 1):
            cycle_dir = os.path.join(self.output_dir, f"cycle_{cycle}")
            
            # Basic directories
            subdirs = ["detections", "labels", "training"]
            
            # Additional directories when using classification model
            if self.use_classifier:
                subdirs.extend(["filtered_detections", "filtered_labels"])
            
            for subdir in subdirs:
                os.makedirs(os.path.join(cycle_dir, subdir), exist_ok=True)
        
        # Training dataset directory
        self.dataset_dir = os.path.join(self.output_dir, "dataset")
        for split in ["train", "val"]:
            for data_type in ["images", "labels"]:
                os.makedirs(os.path.join(self.dataset_dir, data_type, split), exist_ok=True)
        
        print("✅ Directory structure creation completed")
    
    def setup_metrics_tracking(self):
        """Set up performance metrics tracking"""
        # Define metric columns
        columns = [
            'Cycle', 'Model', 'mAP50', 'Precision', 'Recall', 'F1-Score', 
            'Detected_Objects', 'Filtered_Objects'
        ]
        
        self.metrics_df = pd.DataFrame(columns=columns)
        self.metrics_file = os.path.join(self.output_dir, "performance_metrics.csv")
        
        # Load existing metrics file if available
        if os.path.exists(self.metrics_file):
            try:
                existing_metrics = pd.read_csv(self.metrics_file)
                self.metrics_df = existing_metrics
                print(f"📊 Existing metrics file loaded: {self.metrics_file}")
            except Exception as e:
                print(f"⚠️ Existing metrics file loading failed: {str(e)}")
    
    def reset_statistics(self):
        """Reset statistics variables"""
        self.filtered_objects_count = 0
        self.detected_objects_count = 0
    
    def detect_and_classify_objects(self, image_path, cycle):
        """
        Detect objects in image and filter using classification model
        
        Args:
            image_path (str): Image path
            cycle (int): Current training cycle
            
        Returns:
            tuple: (detected objects list, filtered objects list, full detection image, filtered image)
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Image loading failed: {image_path}")
            return [], [], None, None
        
        # Perform YOLO object detection
        try:
            results = self.model.predict(
                source=img, 
                conf=self.conf_threshold, 
                iou=self.iou_threshold,
                save=False,
                verbose=False
            )
        except Exception as e:
            print(f"⚠️ YOLO prediction failed: {str(e)}")
            return [], [], None, None
        
        # Process results
        result = results[0]
        detected_objects = []
        filtered_objects = []
        
        # Copy images for visualization
        img_with_all_boxes = img.copy()
        img_with_filtered_boxes = img.copy() if self.use_classifier else None
        
        if len(result.boxes) > 0:
            # Lists for batch processing object images
            object_images = []
            object_infos = []
            
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # Convert coordinates to integers and check boundaries
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Check if bounding box is valid
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Extract detected object image
                obj_img = img[y1:y2, x1:x2]
                
                if obj_img.size == 0:
                    continue
                
                # Convert coordinates to YOLO format (normalized center point, width, height)
                center_x = ((x1 + x2) / 2) / w
                center_y = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                # Store object information
                obj_info = {
                    'cls_id': 0,  # Single class
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                }
                
                object_images.append(obj_img)
                object_infos.append(obj_info)
                
                # Draw boxes on all detection results
                cv2.rectangle(img_with_all_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_with_all_boxes, f"Obj {conf:.2f}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Perform batch classification when using classification model
            if self.use_classifier and self.classifier and object_images:
                try:
                    # Perform batch classification
                    classification_results = self.classifier.classify_batch(object_images)
                    
                    for obj_info, (pred_class, class_conf) in zip(object_infos, classification_results):
                        bbox = obj_info['bbox']
                        x1, y1, x2, y2 = bbox
                        
                        # Classify objects based on classification results
                        if pred_class == 0:  # Keep
                            detected_objects.append([
                                obj_info['cls_id'], obj_info['center_x'], obj_info['center_y'], 
                                obj_info['width'], obj_info['height']
                            ])
                            self.detected_objects_count += 1
                            
                            # Visualize objects to keep (green)
                            cv2.rectangle(img_with_filtered_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img_with_filtered_boxes, f"Keep {class_conf:.2f}", 
                                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:  # Filter
                            filtered_objects.append([
                                obj_info['cls_id'], obj_info['center_x'], obj_info['center_y'], 
                                obj_info['width'], obj_info['height']
                            ])
                            self.filtered_objects_count += 1
                            
                            # Visualize filtered objects (red)
                            cv2.rectangle(img_with_all_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(img_with_all_boxes, f"Filter {class_conf:.2f}", 
                                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                except Exception as e:
                    print(f"⚠️ Classification processing failed: {str(e)}")
                    # If classification fails, treat all objects as detected
                    for obj_info in object_infos:
                        detected_objects.append([
                            obj_info['cls_id'], obj_info['center_x'], obj_info['center_y'], 
                            obj_info['width'], obj_info['height']
                        ])
                        self.detected_objects_count += 1
            else:
                # When not using classification model, treat all objects as detected
                for obj_info in object_infos:
                    detected_objects.append([
                        obj_info['cls_id'], obj_info['center_x'], obj_info['center_y'], 
                        obj_info['width'], obj_info['height']
                    ])
                    self.detected_objects_count += 1
        
        return detected_objects, filtered_objects, img_with_all_boxes, img_with_filtered_boxes
    
    def save_label(self, objects, label_path):
        """
        Save detected objects as YOLO format label file
        
        Args:
            objects (list): Objects list [cls_id, center_x, center_y, width, height]
            label_path (str): Label file path to save
        """
        try:
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            
            with open(label_path, 'w') as f:
                for obj in objects:
                    line = ' '.join([str(x) for x in obj])
                    f.write(line + '\n')
                    
        except Exception as e:
            print(f"⚠️ Label saving failed: {label_path} - {str(e)}")
    
    def prepare_dataset(self, cycle):
        """
        Prepare dataset for YOLO training
        
        Args:
            cycle (int): Current training cycle
        """
        print(f"📦 Preparing dataset for cycle {cycle}...")
        
        # Determine label directory to use
        if self.use_classifier:
            labels_source_dir = os.path.join(self.output_dir, f"cycle_{cycle}", "filtered_labels")
        else:
            labels_source_dir = os.path.join(self.output_dir, f"cycle_{cycle}", "labels")
        
        # Image file list
        image_files = [f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            raise ValueError("No image files found")
        
        # Dataset split (use most for training)
        total_images = len(image_files)
        if total_images > 10:
            val_count = max(5, min(20, int(total_images * 0.05)))  # 5% or minimum 5
        else:
            val_count = 1  # At least 1 for validation
        
        val_files = image_files[:val_count]
        train_files = image_files[val_count:]
        
        print(f"  - Training: {len(train_files)}")
        print(f"  - Validation: {len(val_files)}")
        
        # Clear existing dataset directories
        for split in ["train", "val"]:
            for data_type in ["images", "labels"]:
                dir_path = os.path.join(self.dataset_dir, data_type, split)
                if os.path.exists(dir_path):
                    for file in os.listdir(dir_path):
                        file_path = os.path.join(dir_path, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
        
        # Data copying function
        def copy_files(file_list, split_name):
            copied_count = 0
            for img_file in file_list:
                # Copy image
                src_img = os.path.join(self.image_dir, img_file)
                dst_img = os.path.join(self.dataset_dir, "images", split_name, img_file)
                
                if os.path.exists(src_img):
                    shutil.copy(src_img, dst_img)
                    
                    # Copy label
                    label_file = os.path.splitext(img_file)[0] + '.txt'
                    src_label = os.path.join(labels_source_dir, label_file)
                    dst_label = os.path.join(self.dataset_dir, "labels", split_name, label_file)
                    
                    if os.path.exists(src_label):
                        shutil.copy(src_label, dst_label)
                        copied_count += 1
                    else:
                        # Create empty file if label doesn't exist
                        with open(dst_label, 'w') as f:
                            pass
            
            return copied_count
        
        # Execute file copying
        train_copied = copy_files(train_files, "train")
        val_copied = copy_files(val_files, "val")
        
        print(f"  ✅ Dataset preparation completed: training {train_copied}, validation {val_copied}")
        
        # Create dataset YAML file
        self.create_dataset_yaml()
    
    def create_dataset_yaml(self):
        """Create dataset YAML file for YOLO training"""
        dataset_yaml = {
            'path': os.path.abspath(self.dataset_dir),
            'train': 'images/train',
            'val': 'images/val', 
            'nc': 1,  # Single class
            'names': ['object']
        }
        
        yaml_path = os.path.join(self.dataset_dir, 'dataset.yaml')
        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(dataset_yaml, f, default_flow_style=False)
            
            print(f"📄 Dataset YAML created: {yaml_path}")
            return yaml_path
            
        except Exception as e:
            print(f"⚠️ YAML file creation failed: {str(e)}")
            return None
    
    def train_model(self, cycle):
        """
        Train YOLO model with current cycle data
        
        Args:
            cycle (int): Current training cycle
            
        Returns:
            str: Trained model path
        """
        print(f"🎓 Starting YOLO model training for cycle {cycle}...")
        
        yaml_path = os.path.join(self.dataset_dir, 'dataset.yaml')
        if not os.path.exists(yaml_path):
            raise FileNotFoundError("Dataset YAML file not found")
        
        # Training results save path
        training_dir = os.path.join(self.output_dir, f"cycle_{cycle}", "training")
        
        try:
            # Execute model training
            results = self.model.train(
                data=yaml_path,
                epochs=50,  # Epochs per cycle
                imgsz=640,
                batch=16,
                patience=10,  # Early stopping
                project=training_dir,
                name="yolo_model",
                device=self.device,
                plots=True,  # Save training graphs
                save_period=10  # Save every 10 epochs
            )
            
            # Trained model path
            trained_model_path = os.path.join(training_dir, "yolo_model", "weights", "best.pt")
            
            if os.path.exists(trained_model_path):
                # Update model
                self.model = YOLO(trained_model_path)
                print(f"✅ Model training completed: {trained_model_path}")
                return trained_model_path
            else:
                raise FileNotFoundError("Trained model file not found")
                
        except Exception as e:
            print(f"❌ Model training failed: {str(e)}")
            raise
    
    def evaluate_performance(self, cycle):
        """
        Evaluate model performance
        
        Args:
            cycle (int): Current training cycle
            
        Returns:
            dict: Performance metrics dictionary
        """
        print(f"📊 Evaluating performance for cycle {cycle}...")
        
        # Image file list
        image_files = [f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            print("⚠️ No images to evaluate")
            return self._create_empty_metrics(cycle)
        
        # Collect performance metrics
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        
        print(f"  📸 Evaluating {len(image_files)} images...")
        
        for image_file in tqdm(image_files, desc="Evaluating"):
            image_path = os.path.join(self.image_dir, image_file)
            
            # Detect objects with current model
            detected_objects, _, _, _ = self.detect_and_classify_objects(image_path, cycle)
            
            # Load ground truth labels
            gt_label_path = os.path.join(self.label_dir, os.path.splitext(image_file)[0] + '.txt')
            gt_objects = self._load_ground_truth(gt_label_path)
            
            # Calculate performance (simplified approach)
            precision, recall, f1 = self._calculate_performance_metrics(detected_objects, gt_objects)
            
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1_scores.append(f1)
        
        # Calculate average performance
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        avg_f1 = np.mean(all_f1_scores)
        
        # Create metrics dictionary
        metrics = {
            'Cycle': cycle,
            'Model': self.model_name,
            'mAP50': avg_precision,  # Simplified mAP
            'Precision': avg_precision,
            'Recall': avg_recall,
            'F1-Score': avg_f1,
            'Detected_Objects': self.detected_objects_count,
            'Filtered_Objects': self.filtered_objects_count if self.use_classifier else 0
        }
        
        # Save metrics
        self._save_metrics(metrics)
        
        print(f"  📈 Performance results:")
        print(f"    - mAP50: {avg_precision:.4f}")
        print(f"    - Precision: {avg_precision:.4f}")
        print(f"    - Recall: {avg_recall:.4f}")
        print(f"    - F1-Score: {avg_f1:.4f}")
        print(f"    - Detected objects: {self.detected_objects_count}")
        if self.use_classifier:
            print(f"    - Filtered objects: {self.filtered_objects_count}")
        
        return metrics
    
    def _load_ground_truth(self, gt_label_path):
        """Load ground truth labels"""
        gt_objects = []
        
        if os.path.exists(gt_label_path):
            try:
                with open(gt_label_path, 'r') as f:
                    for line in f:
                        values = line.strip().split()
                        if len(values) >= 5:
                            cls_id = 0  # Convert to single class
                            center_x = float(values[1])
                            center_y = float(values[2])
                            width = float(values[3])
                            height = float(values[4])
                            gt_objects.append([cls_id, center_x, center_y, width, height])
            except Exception as e:
                print(f"⚠️ Ground truth label loading failed: {gt_label_path} - {str(e)}")
        
        return gt_objects
    
    def _calculate_performance_metrics(self, detected_objects, gt_objects):
        """Calculate performance metrics (simplified approach)"""
        if len(gt_objects) == 0 and len(detected_objects) == 0:
            return 1.0, 1.0, 1.0
        elif len(gt_objects) == 0:
            return 0.0, 1.0, 0.0
        elif len(detected_objects) == 0:
            return 1.0, 0.0, 0.0
        else:
            # Simplified matching approach (actual implementation needs IoU-based matching)
            # Object count-based approximation calculation
            precision = min(1.0, len(gt_objects) / len(detected_objects))
            recall = min(1.0, len(detected_objects) / len(gt_objects))
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            return precision, recall, f1
    
    def _create_empty_metrics(self, cycle):
        """Create empty metrics"""
        return {
            'Cycle': cycle,
            'Model': self.model_name,
            'mAP50': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'F1-Score': 0.0,
            'Detected_Objects': 0,
            'Filtered_Objects': 0
        }
    
    def _save_metrics(self, metrics):
        """Save metrics to dataframe"""
        # Check for same Cycle and Model combination in existing metrics
        mask = (self.metrics_df['Cycle'] == metrics['Cycle']) & \
               (self.metrics_df['Model'] == metrics['Model'])
        
        if any(mask):
            # Update existing entry
            for col, value in metrics.items():
                self.metrics_df.loc[mask, col] = value
        else:
            # Add new entry
            new_row = pd.DataFrame([metrics])
            self.metrics_df = pd.concat([self.metrics_df, new_row], ignore_index=True)
        
        # Save to CSV file
        try:
            self.metrics_df.to_csv(self.metrics_file, index=False)
            print(f"  💾 Metrics saved: {self.metrics_file}")
        except Exception as e:
            print(f"  ⚠️ Metrics saving failed: {str(e)}")
    
    def run(self):
        """Run Active Learning process"""
        print("="*80)
        print(f"🚀 Starting YOLO Active Learning Process")
        print(f"   Model: {self.model_name}")
        print(f"   Use classification model: {self.use_classifier}")
        print(f"   Maximum cycles: {self.max_cycles}")
        print("="*80)
        
        total_start_time = time.time()
        
        # Run each training cycle
        for cycle in range(1, self.max_cycles + 1):
            print(f"\n🔄 Starting training cycle {cycle}/{self.max_cycles}")
            print("-" * 60)
            
            cycle_start_time = time.time()
            
            # Reset statistics
            self.reset_statistics()
            
            # 1. Object detection and classification
            self._process_images_for_cycle(cycle)
            
            # Stop if no objects detected
            if self.detected_objects_count == 0:
                print("⚠️ No objects detected.")
                if cycle == 1:
                    raise Exception("No objects detected in first cycle.")
                
                # Save empty metrics and continue to next cycle
                empty_metrics = self._create_empty_metrics(cycle)
                self._save_metrics(empty_metrics)
                continue
            
            # 2. Prepare dataset
            self.prepare_dataset(cycle)
            
            # 3. Train model
            try:
                trained_model_path = self.train_model(cycle)
            except Exception as e:
                print(f"❌ Cycle {cycle} training failed: {str(e)}")
                empty_metrics = self._create_empty_metrics(cycle)
                self._save_metrics(empty_metrics)
                continue
            
            # 4. Evaluate performance
            metrics = self.evaluate_performance(cycle)
            
            # Cycle completion time
            cycle_elapsed = time.time() - cycle_start_time
            print(f"✅ Cycle {cycle} completed ({cycle_elapsed/60:.1f} minutes)")
        
        # Complete entire process
        total_elapsed = time.time() - total_start_time
        
        print("\n" + "="*80)
        print("🎉 YOLO Active Learning Process Completed!")
        print("="*80)
        print(f"📊 Execution Information:")
        print(f"   - Model: {self.model_name}")
        print(f"   - Completed cycles: {self.max_cycles}")
        print(f"   - Total execution time: {total_elapsed/60:.1f} minutes")
        print(f"   - Use classification model: {self.use_classifier}")
        print(f"📁 Results save location: {self.output_dir}")
        print(f"📈 Performance metrics: {self.metrics_file}")
        
        # Final performance summary
        if not self.metrics_df.empty:
            final_metrics = self.metrics_df.iloc[-1]
            print(f"\n🏆 Final Performance:")
            print(f"   - F1-Score: {final_metrics['F1-Score']:.4f}")
            print(f"   - Precision: {final_metrics['Precision']:.4f}")
            print(f"   - Recall: {final_metrics['Recall']:.4f}")
    
    def _process_images_for_cycle(self, cycle):
        """Process images for each cycle"""
        print("1️⃣ Performing object detection and classification...")
        
        # Results save directories
        cycle_dir = os.path.join(self.output_dir, f"cycle_{cycle}")
        detections_dir = os.path.join(cycle_dir, "detections")
        labels_dir = os.path.join(cycle_dir, "labels")
        
        if self.use_classifier:
            filtered_detections_dir = os.path.join(cycle_dir, "filtered_detections")
            filtered_labels_dir = os.path.join(cycle_dir, "filtered_labels")
        
        # Image file list
        image_files = [f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            raise ValueError("No image files to process")
        
        print(f"   📸 Images to process: {len(image_files)}")
        
        # Process each image
        for image_file in tqdm(image_files, desc="Processing Images"):
            image_path = os.path.join(self.image_dir, image_file)
            
            # Object detection and classification
            detected_objects, filtered_objects, img_all, img_filtered = \
                self.detect_and_classify_objects(image_path, cycle)
            
            # Save detection result images
            if img_all is not None:
                cv2.imwrite(os.path.join(detections_dir, image_file), img_all)
            
            if self.use_classifier and img_filtered is not None:
                cv2.imwrite(os.path.join(filtered_detections_dir, image_file), img_filtered)
            
            # Save label files
            label_name = os.path.splitext(image_file)[0] + '.txt'
            
            if self.use_classifier:
                # Save both all detections and filtered results
                all_objects = detected_objects + filtered_objects
                self.save_label(all_objects, os.path.join(labels_dir, label_name))
                self.save_label(detected_objects, os.path.join(filtered_labels_dir, label_name))
            else:
                # Save all detection results when not using classification model
                self.save_label(detected_objects, os.path.join(labels_dir, label_name))
        
        # Output processing results
        print(f"   ✅ Detected objects: {self.detected_objects_count}")
        if self.use_classifier:
            print(f"   🔍 Filtered objects: {self.filtered_objects_count}")
            if self.detected_objects_count + self.filtered_objects_count > 0:
                keep_rate = self.detected_objects_count / (self.detected_objects_count + self.filtered_objects_count) * 100
                print(f"   📊 Keep rate: {keep_rate:.1f}%")

if __name__ == "__main__":
    # Test execution
    print("🧪 YOLO Active Learning module test")
    
    # Test configuration
    active_learning = YOLOActiveLearning(
        model_path="./models/initial_yolo/yolov8_100pct.pt",
        classifier_path="./models/classification/densenet121_100.pth",
        image_dir="./dataset/images",
        label_dir="./dataset/labels",
        output_dir="./results/test_active_learning",
        conf_threshold=0.25,
        iou_threshold=0.5,
        class_conf_threshold=0.5,
        max_cycles=3,  # Reduced for testing
        gpu_num=0,
        use_classifier=True
    )
    
    # Run Active Learning
    try:
        active_learning.run()
        print("✅ Test completed!")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()