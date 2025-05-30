# main.py (Fixed)
"""
Modified main.py that flexibly adapts to data ratio settings
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

# Module imports
from modules.initial_yolo_trainer import InitialYOLOTrainer
from modules.classification_trainer import ClassificationTrainer
from modules.iterative_processor import IterativeProcessor
from modules.object_classifier import ObjectClassifier
from modules.yolo_active_learning import YOLOActiveLearning

class CompletePipeline:
    """Complete pipeline management class (flexible data ratio support)"""
    
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        
        print("🔧 Pipeline Configuration:")
        print(f"  - Image directory: {config['images_dir']}")
        print(f"  - Label directory: {config['labels_dir']}")
        print(f"  - Results save: {config['iterative_output']}")
        print(f"  - Data ratios: {config['data_percentages']}")
        print(f"  - GPU: {config['gpu_num']}")
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.config['initial_yolo_output'],
            self.config['first_inference_output'],
            self.config['manual_labeling_output'],
            self.config['classification_output'],
            self.config['iterative_output']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def step1_initial_yolo_training(self):
        """Step 1: Initial YOLO model training"""
        print("="*80)
        print("STEP 1: Initial YOLO Model Training")
        print("="*80)
        
        trainer = InitialYOLOTrainer(
            dataset_root=self.config['dataset_root'],
            images_dir=self.config['images_dir'],
            labels_dir=self.config['labels_dir'],
            output_dir=self.config['initial_yolo_output'],
            model_type=self.config['yolo_model_type'],
            epochs=self.config['yolo_epochs'],
            img_size=self.config['img_size'],
            batch_size=self.config['batch_size'],
            percentages=self.config['data_percentages']
        )
        
        trained_models = trainer.train_all_percentages()
        
        # Select model with highest ratio (or maximum value if 100% doesn't exist)
        max_percentage = max(self.config['data_percentages'])
        best_model_path = trained_models.get(max_percentage)
        
        if not best_model_path or not os.path.exists(best_model_path):
            # Select from actually existing models
            available_models = {k: v for k, v in trained_models.items() if v and os.path.exists(v)}
            
            if not available_models:
                raise Exception("No trained YOLO models available.")
            
            # Select model with highest available ratio
            max_available_percentage = max(available_models.keys())
            best_model_path = available_models[max_available_percentage]
            
            print(f"📝 Using {max_available_percentage}% model as {max_percentage}% model is not available.")
        
        print(f"✅ Initial YOLO model training completed: {best_model_path}")
        return best_model_path
    
    def step2_first_inference_and_manual_labeling(self, yolo_model_path):
        """Step 2: First inference with initial model + manual labeling (including GUI alternatives)"""
        print("="*80)
        print("STEP 2: First Inference with Initial Model + Manual Labeling")
        print("="*80)
        
        # Check existing labeling data
        class0_dir = os.path.join(self.config['manual_labeling_output'], 'class0')
        class1_dir = os.path.join(self.config['manual_labeling_output'], 'class1')
        
        if (os.path.exists(class0_dir) and os.path.exists(class1_dir) and 
            len(os.listdir(class0_dir)) > 0 and len(os.listdir(class1_dir)) > 0):
            print("📂 Existing manual labeling data found!")
            print(f"  - Class 0: {len(os.listdir(class0_dir))} items")
            print(f"  - Class 1: {len(os.listdir(class1_dir))} items")
            
            use_existing = input("Would you like to use existing labeling data? (y/n): ").lower().strip()
            if use_existing == 'y':
                return self.config['manual_labeling_output']
        
        # Run first inference
        print("🔍 Running first inference with initial model...")
        self._perform_first_inference(yolo_model_path)
        
        # Select labeling method
        print("\n🏷️ Select manual labeling method:")
        print("1. GUI Labeling (Recommended)")
        print("2. CLI Labeling (Terminal-based)")
        print("3. Batch Labeling (File-based)")
        print("4. Auto Labeling (Confidence-based)")
        
        while True:
            choice = input("Choose (1-4): ").strip()
            
            if choice == '1':
                # Try GUI labeling
                try:
                    return self._try_gui_labeling(yolo_model_path)
                except Exception as e:
                    print(f"❌ GUI labeling failed: {str(e)}")
                    print("Please choose another method.")
                    continue
                    
            elif choice == '2':
                # CLI labeling
                return self._try_cli_labeling(yolo_model_path)
                
            elif choice == '3':
                # Batch labeling
                return self._try_batch_labeling(yolo_model_path)
                
            elif choice == '4':
                # Auto labeling
                return self._try_auto_labeling(yolo_model_path)
                
            else:
                print("Invalid choice. Please select from 1-4.")
    
    def _perform_first_inference(self, yolo_model_path):
        """Perform first inference with initial model"""
        from ultralytics import YOLO
        import cv2
        from tqdm import tqdm
        
        model = YOLO(yolo_model_path)
        inference_dir = self.config['first_inference_output']
        os.makedirs(inference_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(self.config['images_dir']) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"  📸 Running first inference on {len(image_files)} images...")
        
        total_detections = 0
        
        for image_file in tqdm(image_files, desc="First Inference"):
            image_path = os.path.join(self.config['images_dir'], image_file)
            
            results = model.predict(
                source=image_path,
                conf=self.config['conf_threshold'],
                iou=self.config['iou_threshold'],
                save=False,
                verbose=False
            )
            
            img = cv2.imread(image_path)
            if img is not None and len(results[0].boxes) > 0:
                result = results[0]
                
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(img, f"Obj {conf:.2f}", 
                               (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    total_detections += 1
                
                output_path = os.path.join(inference_dir, f"inference_{image_file}")
                cv2.imwrite(output_path, img)
        
        print(f"  ✅ First inference completed: {total_detections} objects detected")
    
    def _try_gui_labeling(self, yolo_model_path):
        """Try GUI labeling"""
        try:
            from modules.manual_labeling_ui import ManualLabelingUI
            
            labeling_ui = ManualLabelingUI(
                yolo_model_path=yolo_model_path,
                images_dir=self.config['images_dir'],
                output_dir=self.config['manual_labeling_output'],
                conf_threshold=self.config['conf_threshold'],
                iou_threshold=self.config['iou_threshold']
            )
            
            labeled_data_path = labeling_ui.run()
            
            if not labeled_data_path:
                raise Exception("GUI labeling was not completed.")
            
            print(f"✅ GUI labeling completed: {labeled_data_path}")
            return labeled_data_path
            
        except ImportError as e:
            if "tkinter" in str(e).lower():
                print("❌ GUI library (tkinter) is not installed.")
                print("Ubuntu/Debian: sudo apt-get install python3-tk")
                print("CentOS/RHEL: sudo yum install tkinter")
            else:
                print(f"❌ GUI module import failed: {str(e)}")
            raise
        except Exception as e:
            print(f"❌ GUI execution failed: {str(e)}")
            raise
    
    def _try_auto_labeling(self, yolo_model_path):
        """Try auto labeling (simple implementation)"""
        print("🤖 Running auto labeling...")
        
        # Set auto labeling threshold
        threshold_input = input("Enter auto classification threshold (0.3-0.9, default: 0.6): ").strip()
        try:
            threshold = float(threshold_input) if threshold_input else 0.6
        except ValueError:
            threshold = 0.6
        
        print(f"  📊 Threshold: {threshold} (≥ {threshold}: Keep, < {threshold}: Filter)")
        
        from ultralytics import YOLO
        import cv2
        from tqdm import tqdm
        
        # Load YOLO model
        model = YOLO(yolo_model_path)
        
        # Output directories
        class0_dir = os.path.join(self.config['manual_labeling_output'], 'class0')
        class1_dir = os.path.join(self.config['manual_labeling_output'], 'class1')
        os.makedirs(class0_dir, exist_ok=True)
        os.makedirs(class1_dir, exist_ok=True)
        
        # Image file list
        image_files = [f for f in os.listdir(self.config['images_dir']) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        class0_count = 0
        class1_count = 0
        
        for image_file in tqdm(image_files, desc="Auto Labeling"):
            image_path = os.path.join(self.config['images_dir'], image_file)
            
            try:
                img = cv2.imread(image_path)
                if img is None:
                    continue
                
                results = model.predict(
                    source=img,
                    conf=self.config['conf_threshold'],
                    iou=self.config['iou_threshold'],
                    save=False,
                    verbose=False
                )
                
                result = results[0]
                
                if len(result.boxes) > 0:
                    for i, box in enumerate(result.boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        # Convert coordinates to integers and check boundaries
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        h, w = img.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        # Check if bounding box is valid
                        if x2 > x1 and y2 > y1:
                            # Extract object image
                            obj_img = img[y1:y2, x1:x2]
                            
                            if obj_img.size > 0:
                                # Auto classification based on confidence
                                if conf >= threshold:
                                    # Class 0 (Keep)
                                    output_dir = class0_dir
                                    class0_count += 1
                                else:
                                    # Class 1 (Filter)
                                    output_dir = class1_dir
                                    class1_count += 1
                                
                                # Generate filename and save
                                base_name = os.path.splitext(image_file)[0]
                                obj_filename = f"{base_name}_obj_{i:03d}_conf_{conf:.3f}.jpg"
                                obj_path = os.path.join(output_dir, obj_filename)
                                
                                cv2.imwrite(obj_path, obj_img)
                                
            except Exception as e:
                print(f"⚠️ Error ({image_file}): {str(e)}")
        
        total_objects = class0_count + class1_count
        
        print(f"\n🎉 Auto labeling completed!")
        print(f"📊 Results:")
        print(f"   Class 0 (Keep): {class0_count} items")
        print(f"   Class 1 (Filter): {class1_count} items")
        print(f"   Total objects: {total_objects} items")
        print(f"💾 Save location: {self.config['manual_labeling_output']}")
        
        if total_objects > 0:
            return self.config['manual_labeling_output']
        else:
            raise Exception("No objects were auto-labeled.")
    
    def _try_cli_labeling(self, yolo_model_path):
        """CLI labeling (simple implementation)"""
        print("💻 CLI labeling is currently replaced with auto labeling.")
        return self._try_auto_labeling(yolo_model_path)
    
    def _try_batch_labeling(self, yolo_model_path):
        """Batch labeling (simple implementation)"""
        print("📁 Batch labeling is currently replaced with auto labeling.")
        return self._try_auto_labeling(yolo_model_path)
    
    def step3_classification_training(self, labeled_data_path):
        """Step 3: Classification model training"""
        print("="*80)
        print("STEP 3: Classification Model Training")
        print("="*80)
        
        trainer = ClassificationTrainer(
            class0_dir=os.path.join(labeled_data_path, 'class0'),
            class1_dir=os.path.join(labeled_data_path, 'class1'),
            output_dir=self.config['classification_output'],
            batch_size=self.config['classification_batch_size'],
            num_epochs=self.config['classification_epochs'],
            gpu_num=self.config['gpu_num']
        )
        
        results = trainer.train_with_data_ratio(
            ratios=self.config['classification_ratios']
        )
        
        # Select model with highest ratio
        max_ratio = max(self.config['classification_ratios'])
        best_classifier_path = os.path.join(
            self.config['classification_output'], 
            f'densenet121_{int(max_ratio*100)}.pth'
        )
        
        if not os.path.exists(best_classifier_path):
            # Find actually existing classification models
            import glob
            available_models = glob.glob(os.path.join(self.config['classification_output'], 'densenet121_*.pth'))
            
            if not available_models:
                raise Exception("Classification model training failed.")
            
            # Select most recent model
            best_classifier_path = max(available_models, key=os.path.getctime)
            print(f"📝 Using {os.path.basename(best_classifier_path)} as {max_ratio*100}% model is not available.")
        
        print(f"✅ Classification model training completed: {best_classifier_path}")
        return best_classifier_path
    
    def step4_iterative_process(self, classifier_path):
        """Step 4: Run Iterative Process"""
        print("="*80)
        print("STEP 4: Iterative Active Learning Process")
        print("="*80)
        
        processor = IterativeProcessor(
            yolo_models_dir=self.config['initial_yolo_output'],
            classifier_path=classifier_path,
            image_dir=self.config['images_dir'],
            label_dir=self.config['labels_dir'],
            output_dir=self.config['iterative_output'],
            conf_threshold=self.config['conf_threshold'],
            iou_threshold=self.config['iou_threshold'],
            class_conf_threshold=self.config['class_conf_threshold'],
            max_cycles=self.config['max_cycles'],
            gpu_num=self.config['gpu_num']
        )
        
        results = processor.run_iterative_experiments()
        
        print("✅ Iterative Process completed!")
        return results
    
    def run_complete_pipeline(self):
        """Run complete pipeline"""
        total_start_time = time.time()
        
        try:
            print("🚀 Starting Complete Pipeline!")
            print("="*80)
            
            # Step 1: Initial YOLO training
            yolo_model_path = self.step1_initial_yolo_training()
            
            # Step 2: First inference + manual labeling
            labeled_data_path = self.step2_first_inference_and_manual_labeling(yolo_model_path)
            
            # Step 3: Classification training
            classifier_path = self.step3_classification_training(labeled_data_path)
            
            # Step 4: Iterative Process
            final_results = self.step4_iterative_process(classifier_path)
            
            # Total execution time
            total_elapsed = time.time() - total_start_time
            
            print("="*80)
            print("🎉 Complete Pipeline Finished!")
            print("="*80)
            print(f"⏰ Total execution time: {total_elapsed/60:.1f} minutes")
            print(f"📊 Results summary:")
            print(f"  - Initial YOLO model: {yolo_model_path}")
            print(f"  - Manual labeling data: {labeled_data_path}")
            print(f"  - Classification model: {classifier_path}")
            print(f"  - Final results: {self.config['iterative_output']}")
            
            if final_results and "summary" in final_results:
                summary = final_results["summary"]
                print(f"📈 Iterative Process results:")
                print(f"  - Success rate: {summary['success_rate']:.1f}%")
                print(f"  - Processed models: {summary['total_models']}")
            
            return final_results
            
        except Exception as e:
            print(f"❌ Error occurred during pipeline execution: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_single_step(self, step_num):
        """Run specific step only"""
        if step_num == 1:
            return self.step1_initial_yolo_training()
        elif step_num == 2:
            # Find available YOLO models
            import glob
            model_files = glob.glob(os.path.join(self.config['initial_yolo_output'], 'yolov8_*.pt'))
            
            if not model_files:
                print(f"❌ No YOLO models found: {self.config['initial_yolo_output']}")
                print("Please run Step 1 first.")
                return None
            
            # Select most recent model or highest ratio model
            yolo_model = max(model_files, key=os.path.getctime)
            print(f"📝 Using YOLO model: {os.path.basename(yolo_model)}")
            
            return self.step2_first_inference_and_manual_labeling(yolo_model)
        elif step_num == 3:
            labeled_data = self.config['manual_labeling_output']
            if not os.path.exists(os.path.join(labeled_data, 'class0')):
                print(f"❌ Manual labeling data not found: {labeled_data}")
                print("Please run Step 2 first.")
                return None
            return self.step3_classification_training(labeled_data)
        elif step_num == 4:
            # Find available classification models
            import glob
            classifier_files = glob.glob(os.path.join(self.config['classification_output'], 'densenet121_*.pth'))
            
            if not classifier_files:
                print(f"❌ No classification models found: {self.config['classification_output']}")
                print("Please run Step 3 first.")
                return None
            
            # Select most recent model
            classifier = max(classifier_files, key=os.path.getctime)
            print(f"📝 Using Classification model: {os.path.basename(classifier)}")
            
            return self.step4_iterative_process(classifier)
        else:
            print(f"❌ Invalid step number: {step_num}")
            return None

def create_default_config():
    """Create default configuration"""
    return {
        'dataset_root': './dataset',
        'images_dir': './dataset/images',
        'labels_dir': './dataset/labels',
        'initial_yolo_output': './results/01_initial_yolo',
        'first_inference_output': './results/02_first_inference',
        'manual_labeling_output': './results/03_manual_labeling',
        'classification_output': './results/04_classification',
        'iterative_output': './results/05_iterative_process',
        'yolo_model_type': 'yolov8n.pt',
        'yolo_epochs': 100,
        'img_size': 640,
        'batch_size': 16,
        'data_percentages': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'classification_batch_size': 16,
        'classification_epochs': 30,
        'classification_ratios': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'conf_threshold': 0.25,
        'iou_threshold': 0.5,
        'class_conf_threshold': 0.5,
        'max_cycles': 10,
        'gpu_num': 0,
    }

def load_config(config_path):
    """Load configuration file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ Configuration file loaded: {config_path}")
        return config
    except Exception as e:
        print(f"❌ Configuration file loading failed: {e}")
        return None

def save_config(config, config_path):
    """Save configuration file"""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✅ Configuration file saved: {config_path}")
    except Exception as e:
        print(f"❌ Configuration file saving failed: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Complete Pipeline for YOLO + Classification Active Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  python main.py                              # Run complete pipeline
  python main.py --step 1                     # YOLO training only
  python main.py --step 2                     # First inference + manual labeling only
  python main.py --step 3                     # Classification training only
  python main.py --step 4                     # Iterative Process only
  python main.py --config my_config.json      # Use configuration file
  python main.py --create-config config.json  # Create default configuration file

Labeling Methods:
  1. GUI Labeling: Intuitive graphical interface (Recommended)
  2. CLI Labeling: Interactive labeling in terminal
  3. Batch Labeling: Manual classification using file explorer
  4. Auto Labeling: Automatic classification based on confidence
        """
    )
    
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4], 
                       help='Step to execute (1: YOLO training, 2: Manual labeling, 3: Classification, 4: Iterative)')
    parser.add_argument('--images_dir', type=str, help='Image directory path')
    parser.add_argument('--labels_dir', type=str, help='Label directory path')
    parser.add_argument('--output_dir', type=str, help='Output directory path')
    parser.add_argument('--gpu_num', type=int, help='GPU number to use')
    parser.add_argument('--create-config', type=str, help='Create default configuration file at specified path')
    
    args = parser.parse_args()
    
    # Create default configuration file
    if args.create_config:
        config = create_default_config()
        save_config(config, args.create_config)
        print(f"✅ Default configuration file created: {args.create_config}")
        print("Please modify the configuration and run again.")
        return
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        if config is None:
            return
    else:
        config = create_default_config()
        if args.config:
            print(f"⚠️ Configuration file not found: {args.config}")
            print("Using default configuration.")
    
    # Override configuration with command line arguments
    if args.images_dir:
        config['images_dir'] = args.images_dir
    if args.labels_dir:
        config['labels_dir'] = args.labels_dir
    if args.output_dir:
        config['iterative_output'] = args.output_dir
    if args.gpu_num is not None:
        config['gpu_num'] = args.gpu_num
    
    # Input validation
    if not os.path.exists(config['images_dir']):
        print(f"❌ Image directory not found: {config['images_dir']}")
        return
    
    if not os.path.exists(config['labels_dir']):
        print(f"❌ Label directory not found: {config['labels_dir']}")
        return
    
    # Run pipeline
    pipeline = CompletePipeline(config)
    
    if args.step:
        # Run specific step only
        print(f"🎯 Running Step {args.step}")
        result = pipeline.run_single_step(args.step)
        if result:
            print(f"✅ Step {args.step} completed")
        else:
            print(f"❌ Step {args.step} failed")
    else:
        # Run complete pipeline
        print("🚀 Running complete pipeline")
        result = pipeline.run_complete_pipeline()
        if result:
            print("✅ Complete pipeline finished")
        else:
            print("❌ Pipeline execution failed")

if __name__ == "__main__":
    main()