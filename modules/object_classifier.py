# modules/object_classifier.py
"""
Object Classification Model Module
Module for classifying detected objects using trained DenseNet model
"""

import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class ObjectClassifier:
    """
    Object Classifier class
    Classifies objects detected by YOLO into Class 0 (Keep) or Class 1 (Filter)
    """
    
    def __init__(self, model_path, device=None, conf_threshold=0.5, gpu_num=0):
        """
        Initialize object classification model
        
        Args:
            model_path (str): Pre-trained classification model path (.pth file)
            device (torch.device): Computing device (auto-select if None)
            conf_threshold (float): Classification confidence threshold
            gpu_num (int): GPU number to use
        """
        # Device setup
        if device is None:
            self.device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        
        print(f"🔧 Classification model device: {self.device}")
        print(f"📁 Model path: {model_path}")
        print(f"🎯 Confidence threshold: {conf_threshold}")
        
        # Create model structure
        self.model = self._create_model()
        
        # Load model weights
        self._load_model_weights()
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Set up image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # DenseNet input size
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        print("✅ Object classifier initialization completed")
        
    def _create_model(self):
        """Create DenseNet121 model structure"""
        print("🏗️ Creating DenseNet121 model structure...")
        
        # Initialize DenseNet121 model (without pre-trained weights)
        model = models.densenet121(pretrained=False)
        
        # Modify last layer for binary classification
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Dropout layer
            nn.Linear(num_features, 2)  # 2-class output (0: Keep, 1: Filter)
        )
        
        model = model.to(self.device)
        print("✅ Model structure creation completed")
        
        return model
        
    def _load_model_weights(self):
        """Load model weights"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            print("📥 Loading model weights...")
            
            # Load weights
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            print("✅ Model weights loading successful")
            
        except RuntimeError as e:
            print(f"⚠️ Model weights loading failed: {e}")
            print("🔄 Attempting key mapping...")
            
            try:
                # Remove 'module.' prefix if saved with DataParallel
                new_state_dict = {}
                for key, value in state_dict.items():
                    if 'module.' in key:
                        key = key.replace('module.', '')
                    new_state_dict[key] = value
                
                self.model.load_state_dict(new_state_dict)
                print("✅ Model weights loading successful after key mapping")
                
            except Exception as e2:
                print(f"❌ Final model loading failure: {e2}")
                raise Exception(f"Model loading failed: {e2}")
    
    def preprocess_image(self, image):
        """
        Image preprocessing
        
        Args:
            image (numpy.ndarray): OpenCV format image (BGR)
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Check image size
        if image.shape[0] < 10 or image.shape[1] < 10:
            print("⚠️ Image too small (less than 10x10)")
            return None
        
        # Check if image is empty
        if image.size == 0:
            print("⚠️ Empty image")
            return None
        
        try:
            # Convert OpenCV image (BGR) to PIL image (RGB)
            if len(image.shape) == 3:
                # Color image
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                # Grayscale image
                pil_image = Image.fromarray(image).convert('RGB')
            
            # Apply preprocessing
            input_tensor = self.transform(pil_image)
            
            # Add batch dimension [C, H, W] -> [1, C, H, W]
            input_tensor = input_tensor.unsqueeze(0)
            
            return input_tensor.to(self.device)
            
        except Exception as e:
            print(f"⚠️ Image preprocessing failed: {str(e)}")
            return None
    
    def classify(self, image):
        """
        Perform object image classification
        
        Args:
            image (numpy.ndarray): Object image to classify (OpenCV format, BGR)
            
        Returns:
            tuple: (predicted class, confidence)
                - predicted class: 0 (Keep) or 1 (Filter)
                - confidence: probability value between 0.0 ~ 1.0
        """
        # Image preprocessing
        input_tensor = self.preprocess_image(image)
        
        if input_tensor is None:
            # Return default value (filtering) if preprocessing fails
            return 1, 0.0
        
        try:
            # Perform prediction (disable gradient computation)
            with torch.no_grad():
                # Model forward pass
                outputs = self.model(input_tensor)
                
                # Apply Softmax to calculate probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Select maximum probability and corresponding class
                confidence, predicted_class = torch.max(probabilities, 1)
                
                # Move to CPU and convert to numpy values
                predicted_class = predicted_class.item()
                confidence = confidence.item()
                
                return predicted_class, confidence
                
        except Exception as e:
            print(f"⚠️ Classification prediction failed: {str(e)}")
            # Return default value (filtering) on error
            return 1, 0.0
    
    def classify_batch(self, images):
        """
        Classify multiple images in batch
        
        Args:
            images (list): List of OpenCV images
            
        Returns:
            list: List of results in [(predicted class, confidence), ...] format
        """
        if not images:
            return []
        
        print(f"🔍 Starting batch classification: {len(images)} images")
        
        # Prepare batch tensors
        batch_tensors = []
        valid_indices = []
        
        for i, image in enumerate(images):
            tensor = self.preprocess_image(image)
            if tensor is not None:
                batch_tensors.append(tensor.squeeze(0))  # Remove batch dimension
                valid_indices.append(i)
        
        if not batch_tensors:
            print("⚠️ No valid images")
            return [(1, 0.0)] * len(images)
        
        try:
            # Create batch tensor
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Batch prediction
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidences, predicted_classes = torch.max(probabilities, 1)
                
                # Move to CPU
                predicted_classes = predicted_classes.cpu().numpy()
                confidences = confidences.cpu().numpy()
            
            # Map results
            results = [(1, 0.0)] * len(images)  # Initialize with default values
            
            for i, valid_idx in enumerate(valid_indices):
                results[valid_idx] = (int(predicted_classes[i]), float(confidences[i]))
            
            print(f"✅ Batch classification completed")
            return results
            
        except Exception as e:
            print(f"⚠️ Batch classification failed: {str(e)}")
            return [(1, 0.0)] * len(images)
    
    def get_class_name(self, class_id):
        """
        Convert class ID to name
        
        Args:
            class_id (int): Class ID (0 or 1)
            
        Returns:
            str: Class name
        """
        class_names = {0: "Keep", 1: "Filter"}
        return class_names.get(class_id, "Unknown")
    
    def evaluate_confidence(self, confidence):
        """
        Evaluate confidence level
        
        Args:
            confidence (float): Confidence value
            
        Returns:
            str: Confidence level ("High", "Medium", "Low")
        """
        if confidence >= 0.8:
            return "High"
        elif confidence >= 0.6:
            return "Medium"
        else:
            return "Low"
    
    def classify_with_details(self, image):
        """
        Perform classification with detailed information
        
        Args:
            image (numpy.ndarray): Object image to classify
            
        Returns:
            dict: Detailed classification results
        """
        predicted_class, confidence = self.classify(image)
        
        result = {
            'predicted_class': predicted_class,
            'class_name': self.get_class_name(predicted_class),
            'confidence': confidence,
            'confidence_level': self.evaluate_confidence(confidence),
            'should_keep': predicted_class == 0,
            'above_threshold': confidence >= self.conf_threshold
        }
        
        return result
    
    def filter_objects(self, detected_objects, images):
        """
        Classify and filter detected objects
        
        Args:
            detected_objects (list): List of detected object information
            images (list): List of object images
            
        Returns:
            tuple: (objects to keep, filtered objects)
        """
        keep_objects = []
        filter_objects = []
        
        print(f"🔍 Starting object filtering: {len(detected_objects)} objects")
        
        # Perform batch classification
        classification_results = self.classify_batch(images)
        
        for i, (obj_info, (pred_class, confidence)) in enumerate(zip(detected_objects, classification_results)):
            classification_detail = {
                'index': i,
                'predicted_class': pred_class,
                'class_name': self.get_class_name(pred_class),
                'confidence': confidence,
                'above_threshold': confidence >= self.conf_threshold,
                'object_info': obj_info
            }
            
            if pred_class == 0:  # Keep
                keep_objects.append(classification_detail)
            else:  # Filter
                filter_objects.append(classification_detail)
        
        print(f"📊 Filtering results:")
        print(f"  - Keep: {len(keep_objects)} objects")
        print(f"  - Filter: {len(filter_objects)} objects")
        
        return keep_objects, filter_objects
    
    def save_classification_results(self, results, save_path):
        """
        Save classification results to file
        
        Args:
            results (list): Classification results list
            save_path (str): Save path
        """
        try:
            import json
            
            # Convert results to JSON serializable format
            serializable_results = []
            for result in results:
                if isinstance(result, dict):
                    serializable_results.append(result)
                else:
                    # Convert tuple to dict
                    serializable_results.append({
                        'predicted_class': result[0],
                        'confidence': result[1],
                        'class_name': self.get_class_name(result[0])
                    })
            
            with open(save_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"💾 Classification results saved: {save_path}")
            
        except Exception as e:
            print(f"⚠️ Results saving failed: {str(e)}")
    
    def get_model_info(self):
        """
        Return model information
        
        Returns:
            dict: Model information
        """
        # Calculate number of model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'confidence_threshold': self.conf_threshold,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_architecture': 'DenseNet121',
            'num_classes': 2,
            'class_names': ['Keep', 'Filter']
        }
    
    def __str__(self):
        """String representation"""
        info = self.get_model_info()
        return f"""ObjectClassifier(
    Model: {info['model_architecture']}
    Device: {info['device']}
    Classes: {info['num_classes']} ({', '.join(info['class_names'])})
    Confidence Threshold: {info['confidence_threshold']}
    Parameters: {info['total_parameters']:,}
    Model Path: {info['model_path']}
)"""

# Test and utility functions
def test_classifier(model_path, test_images_dir, output_dir=None):
    """
    Classifier test function
    
    Args:
        model_path (str): Classification model path
        test_images_dir (str): Test image directory
        output_dir (str): Results save directory
    """
    print("🧪 Starting classifier test")
    
    # Initialize classifier
    classifier = ObjectClassifier(model_path)
    print(classifier)
    
    # Load test images
    test_images = []
    image_paths = []
    
    for filename in os.listdir(test_images_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(test_images_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                test_images.append(img)
                image_paths.append(img_path)
    
    if not test_images:
        print("❌ No test images found")
        return
    
    print(f"📸 {len(test_images)} test images loaded")
    
    # Perform classification
    results = []
    for i, (img, img_path) in enumerate(zip(test_images, image_paths)):
        result = classifier.classify_with_details(img)
        result['image_path'] = img_path
        result['filename'] = os.path.basename(img_path)
        results.append(result)
        
        print(f"📊 {i+1:3d}. {result['filename']:20s} -> "
              f"{result['class_name']:6s} ({result['confidence']:.3f}, {result['confidence_level']})")
    
    # Output statistics
    keep_count = sum(1 for r in results if r['predicted_class'] == 0)
    filter_count = len(results) - keep_count
    avg_confidence = np.mean([r['confidence'] for r in results])
    
    print(f"\n📈 Test result statistics:")
    print(f"  - Keep: {keep_count} ({keep_count/len(results)*100:.1f}%)")
    print(f"  - Filter: {filter_count} ({filter_count/len(results)*100:.1f}%)")
    print(f"  - Average confidence: {avg_confidence:.3f}")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        classifier.save_classification_results(results, 
                                             os.path.join(output_dir, 'test_results.json'))

if __name__ == "__main__":
    # Test execution example
    print("🔍 ObjectClassifier module test")
    
    # Set model path (needs to be changed to actual path)
    model_path = "./models/classification/densenet121_100.pth"
    test_images_dir = "./test_images"
    output_dir = "./test_results"
    
    if os.path.exists(model_path):
        test_classifier(model_path, test_images_dir, output_dir)
    else:
        print(f"❌ Model file not found: {model_path}")
        print("Please train the classification model first.")