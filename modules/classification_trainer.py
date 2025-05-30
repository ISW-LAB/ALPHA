# modules/classification_trainer.py
"""
Classification Model Training Module
Module for training DenseNet classification models with manually labeled object data
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import random
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import json

class ObjectDataset(Dataset):
    """Object image dataset class"""
    
    def __init__(self, class0_dir, class1_dir, transform=None):
        """
        Initialize object image dataset
        
        Args:
            class0_dir (str): Class 0 image directory path
            class1_dir (str): Class 1 image directory path
            transform: Image transformer
        """
        self.transform = transform
        self.class_dirs = [class0_dir, class1_dir]
        self.samples = []
        
        # Load images from each class directory
        for class_idx, class_dir in enumerate(self.class_dirs):
            if not os.path.isdir(class_dir):
                print(f"⚠️ Warning: Directory {class_dir} does not exist.")
                continue
                
            image_count = 0
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    try:
                        # Check if image file is valid
                        with Image.open(img_path) as img:
                            img.verify()
                        self.samples.append((img_path, class_idx))
                        image_count += 1
                    except Exception as e:
                        print(f"⚠️ Skipping corrupted image file: {img_path} - {str(e)}")
            
            print(f"Class {class_idx}: {image_count} images loaded ({class_dir})")
        
        print(f"📊 Total {len(self.samples)} images loaded.")
        
        if len(self.samples) == 0:
            raise ValueError("No images loaded. Please check directory paths.")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get item from dataset
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: (image, class label)
        """
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Image loading failed: {img_path} - {str(e)}")
            # Return empty image on error
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class ClassificationTrainer:
    """Classification model trainer"""
    
    def __init__(self, class0_dir, class1_dir, output_dir, batch_size=16, 
                 num_epochs=30, gpu_num=0, random_seed=13):
        """
        Initialize Classification trainer
        
        Args:
            class0_dir (str): Class 0 image directory
            class1_dir (str): Class 1 image directory
            output_dir (str): Results save directory
            batch_size (int): Batch size
            num_epochs (int): Number of training epochs
            gpu_num (int): GPU number to use
            random_seed (int): Random seed
        """
        self.class0_dir = class0_dir
        self.class1_dir = class1_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.gpu_num = gpu_num
        self.random_seed = random_seed
        
        # Set seed for reproducibility
        self.set_seed()
        
        # Device setup
        self.device = torch.device(f"cuda:{self.gpu_num}" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {self.device}")
        
        # Create results save directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define image transformers
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
                transforms.RandomRotation(degrees=10),    # Rotation augmentation
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color augmentation
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        print("✅ Classification Trainer initialization completed")
    
    def set_seed(self):
        """Set seed for reproducibility"""
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def init_weights(self, m):
        """Weight initialization function"""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def create_model(self):
        """Create DenseNet121 model"""
        print("🏗️ Creating DenseNet121 model...")
        
        # Initialize DenseNet121 model (use pre-trained weights)
        model = models.densenet121(pretrained=True)
        
        # Freeze most feature extraction layers (prevent overfitting)
        for param in model.features.parameters():
            param.requires_grad = False
            
        # Make only last dense block trainable
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
            
        # Make last normalization layer trainable
        for param in model.features.norm5.parameters():
            param.requires_grad = True
        
        # Replace classifier layer (include dropout)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Prevent overfitting
            nn.Linear(num_features, 2)  # Binary classification
        )
        
        # Initialize weights for newly added layers
        model.classifier.apply(self.init_weights)
        
        model = model.to(self.device)
        print("✅ Model creation and GPU loading completed")
        
        return model
    
    def train_model(self, model, dataloaders, criterion, optimizer, scheduler, ratio, 
                   metric='val_f1', patience=10):
        """
        Model training function
        
        Args:
            model: Model to train
            dataloaders: Data loader dictionary
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            ratio: Current data ratio being used
            metric: Model selection criterion metric
            patience: Early stopping patience
            
        Returns:
            tuple: (training history, best performance, best performance epoch)
        """
        print(f"🚀 Starting model training (Data ratio: {ratio*100:.0f}%)")
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_precision': [], 'val_recall': [], 'val_f1': []
        }
        
        start_time = time.time()
        best_model_wts = model.state_dict()
        best_score = 0.0
        best_epoch = 0
        early_stop_counter = 0
        
        print(f"📊 Training configuration:")
        print(f"  - Epochs: {self.num_epochs}")
        print(f"  - Early stopping criterion: {metric}")
        print(f"  - Early stopping patience: {patience}")
        
        for epoch in range(self.num_epochs):
            print(f'\n📅 Epoch {epoch+1}/{self.num_epochs}')
            print('-' * 50)
            
            # Each epoch has training and validation phases
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Training mode
                else:
                    model.eval()   # Evaluation mode
                    
                running_loss = 0.0
                all_preds = []
                all_labels = []
                
                # tqdm for progress display
                pbar = tqdm(dataloaders[phase], desc=f'{phase.upper()}')
                
                # Iterate through data
                for inputs, labels in pbar:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Zero parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        # Backward pass + optimize only in training phase
                        if phase == 'train':
                            loss.backward()
                            # Gradient clipping (prevent overfitting)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            
                    # Collect statistics
                    running_loss += loss.item() * inputs.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    # Update progress display
                    pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                
                # Calculate epoch performance
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = accuracy_score(all_labels, all_preds)
                
                print(f'{phase.upper()} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
                
                # Calculate additional metrics in validation phase
                if phase == 'val':
                    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
                    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
                    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
                    
                    print(f'VAL - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
                    
                    # Save history
                    history['val_loss'].append(epoch_loss)
                    history['val_acc'].append(epoch_acc)
                    history['val_precision'].append(precision)
                    history['val_recall'].append(recall)
                    history['val_f1'].append(f1)
                    
                    # Update learning rate scheduler
                    if scheduler is not None:
                        scheduler.step(epoch_loss)
                        current_lr = optimizer.param_groups[0]['lr']
                        print(f'📉 Current learning rate: {current_lr:.6f}')
                    
                    # Check best performance model
                    if metric == 'val_acc':
                        current_score = epoch_acc
                    elif metric == 'val_precision':
                        current_score = precision
                    elif metric == 'val_recall':
                        current_score = recall
                    elif metric == 'val_f1':
                        current_score = f1
                    else:
                        current_score = f1  # Default value
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_epoch = epoch
                        best_model_wts = model.state_dict().copy()
                        print(f"🎉 New best performance! (Epoch {epoch+1}, {metric} = {current_score:.4f})")
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                        print(f"⏳ No performance improvement: {early_stop_counter}/{patience}")
                        
                        if early_stop_counter >= patience:
                            print(f"🛑 Early stopping: {patience} consecutive epochs without improvement")
                            break
                else:
                    # Training phase history
                    history['train_loss'].append(epoch_loss)
                    history['train_acc'].append(epoch_acc)
            
            # Check early stopping
            if early_stop_counter >= patience:
                print(f"🏁 Training early stopped at epoch {epoch+1}.")
                break
        
        # Training completion information
        time_elapsed = time.time() - start_time
        print(f'\n⏰ Training completed: {time_elapsed // 60:.0f}min {time_elapsed % 60:.0f}sec')
        print(f'🏆 Best performance: Epoch {best_epoch+1}, {metric} = {best_score:.4f}')
        
        # Save best performance model
        model_save_path = os.path.join(self.output_dir, f'densenet121_{int(ratio*100)}.pth')
        torch.save(best_model_wts, model_save_path)
        print(f"💾 Best performance model saved: {model_save_path}")
        
        # Save training history
        history_save_path = os.path.join(self.output_dir, f'training_history_{int(ratio*100)}.json')
        with open(history_save_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_history = {k: [float(x) for x in v] for k, v in history.items()}
            json.dump(json_history, f, indent=2)
        print(f"📊 Training history saved: {history_save_path}")
        
        # Restore model to best performance state
        model.load_state_dict(best_model_wts)
        
        return history, best_score, best_epoch+1
    
    def plot_training_history(self, history, ratio):
        """Visualize training history"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training History - {int(ratio*100)}% Data', fontsize=16)
            
            # Loss graph
            axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
            axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Accuracy graph
            axes[0, 1].plot(history['train_acc'], label='Train Acc', color='blue')
            axes[0, 1].plot(history['val_acc'], label='Val Acc', color='red')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Precision, Recall, F1 graph
            axes[1, 0].plot(history['val_precision'], label='Precision', color='green')
            axes[1, 0].plot(history['val_recall'], label='Recall', color='orange')
            axes[1, 0].plot(history['val_f1'], label='F1-Score', color='purple')
            axes[1, 0].set_title('Validation Metrics')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Final performance display
            final_text = f"""Final Performance:
            Accuracy: {history['val_acc'][-1]:.4f}
            Precision: {history['val_precision'][-1]:.4f}
            Recall: {history['val_recall'][-1]:.4f}
            F1-Score: {history['val_f1'][-1]:.4f}"""
            
            axes[1, 1].text(0.1, 0.5, final_text, fontsize=12, verticalalignment='center')
            axes[1, 1].set_title('Final Performance')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, f'training_plot_{int(ratio*100)}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Training graph saved: {plot_path}")
            
        except Exception as e:
            print(f"⚠️ Graph saving failed: {str(e)}")
    
    def train_with_data_ratio(self, ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
                             metric='val_f1'):
        """
        Train model with adjusted data ratios
        
        Args:
            ratios: List of data ratios to use
            metric: Model selection criterion metric
            
        Returns:
            dict: Training results dictionary
        """
        print("="*80)
        print("🎯 Starting Classification Model Training")
        print("="*80)
        
        # Load full dataset
        print("📂 Loading dataset...")
        try:
            full_dataset = ObjectDataset(self.class0_dir, self.class1_dir, 
                                       transform=self.data_transforms['train'])
        except Exception as e:
            print(f"❌ Dataset loading failed: {str(e)}")
            return {}
        
        # Extract class-wise indices
        class_indices = {0: [], 1: []}
        for i, (_, label) in enumerate(full_dataset.samples):
            class_indices[label].append(i)
        
        print(f"📊 Class-wise data distribution:")
        print(f"  - Class 0: {len(class_indices[0])} images")
        print(f"  - Class 1: {len(class_indices[1])} images")
        
        # Warning if data is too small
        total_samples = len(class_indices[0]) + len(class_indices[1])
        if total_samples < 100:
            print("⚠️ Warning: Very small dataset (less than 100). Additional data collection recommended.")
        
        val_ratio = 0.15  # Validation set ratio
        results = {'ratios': [], 'best_scores': [], 'best_epochs': [], 'histories': []}
        
        print(f"\n🔄 Data ratios to train: {ratios}")
        print(f"🎯 Model selection criterion: {metric}")
        
        # Execute training for each data ratio
        for i, ratio in enumerate(ratios):
            print(f"\n{'='*60}")
            print(f"📊 Data ratio: {ratio*100:.0f}% ({i+1}/{len(ratios)})")
            print(f"{'='*60}")
            
            try:
                # Select data by ratio from each class
                train_indices = []
                for label in [0, 1]:
                    class_size = len(class_indices[label])
                    num_samples = int(class_size * ratio)
                    
                    if num_samples < 3:
                        print(f"⚠️ Warning: Too few data points for class {label} ({num_samples})")
                    
                    # Stratified selection (always same order)
                    train_indices.extend(class_indices[label][:num_samples])
                
                # Training/validation split
                val_size = max(int(len(train_indices) * val_ratio), 2)  # At least 2 for validation
                train_size = len(train_indices) - val_size
                
                # Shuffle indices (reproducibly)
                random.shuffle(train_indices)
                train_subset = Subset(full_dataset, train_indices[val_size:])
                
                # Validation dataset (apply separate transformation)
                val_dataset = ObjectDataset(self.class0_dir, self.class1_dir, 
                                          transform=self.data_transforms['val'])
                val_subset = Subset(val_dataset, train_indices[:val_size])
                
                print(f"📊 Data split:")
                print(f"  - Training set: {len(train_subset)}")
                print(f"  - Validation set: {len(val_subset)}")
                
                # Adjust batch size (when data is small)
                actual_batch_size = min(self.batch_size, len(train_subset) // 2)
                actual_batch_size = max(actual_batch_size, 1)  # At least 1
                
                # Create data loaders
                dataloaders = {
                    'train': DataLoader(train_subset, batch_size=actual_batch_size, 
                                      shuffle=True, num_workers=4, pin_memory=True),
                    'val': DataLoader(val_subset, batch_size=actual_batch_size, 
                                    shuffle=False, num_workers=4, pin_memory=True)
                }
                
                print(f"⚙️ Actual batch size: {actual_batch_size}")
                
                # Create model
                model = self.create_model()
                
                # Calculate weights for class imbalance handling
                class_samples = [len(class_indices[0]), len(class_indices[1])]
                if min(class_samples) > 0:
                    weights = [len(full_dataset) / (2 * count) if count > 0 else 1.0 
                             for count in class_samples]
                    class_weights = torch.FloatTensor(weights).to(self.device)
                    criterion = nn.CrossEntropyLoss(weight=class_weights)
                    print(f"⚖️ Class weights applied: {[f'{w:.3f}' for w in weights]}")
                else:
                    criterion = nn.CrossEntropyLoss()
                
                # Optimizer setup
                trainable_params = [param for param in model.parameters() if param.requires_grad]
                optimizer = optim.Adam(
                    trainable_params, 
                    lr=0.0005,  # Low learning rate
                    weight_decay=0.0001  # L2 regularization
                )
                
                # Learning rate scheduler
                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,
                    patience=5,
                    verbose=True,
                    min_lr=1e-7
                )
                
                # Train model
                history, best_score, best_epoch = self.train_model(
                    model, dataloaders, criterion, optimizer, scheduler, 
                    ratio, metric, patience=10
                )
                
                # Save results
                results['ratios'].append(ratio)
                results['best_scores'].append(best_score)
                results['best_epochs'].append(best_epoch)
                results['histories'].append(history)
                
                # Save training graph
                self.plot_training_history(history, ratio)
                
                print(f"✅ {ratio*100:.0f}% data training completed:")
                print(f"   - Best performance ({metric}): {best_score:.4f} (epoch {best_epoch})")
                
            except Exception as e:
                print(f"❌ {ratio*100:.0f}% data training failed: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Record failure in results (with 0)
                results['ratios'].append(ratio)
                results['best_scores'].append(0.0)
                results['best_epochs'].append(0)
                results['histories'].append({})
        
        # Overall results summary
        print("\n" + "="*80)
        print("🏆 Classification Training Results Summary")
        print("="*80)
        
        for i, ratio in enumerate(results['ratios']):
            if i < len(results['best_scores']):
                score = results['best_scores'][i]
                epoch = results['best_epochs'][i]
                status = "✅" if score > 0 else "❌"
                print(f"{status} Data ratio {ratio*100:3.0f}%: {metric} = {score:.4f} (epoch {epoch})")
        
        # Best performance model information
        if results['best_scores']:
            best_idx = np.argmax(results['best_scores'])
            best_ratio = results['ratios'][best_idx]
            best_performance = results['best_scores'][best_idx]
            print(f"\n🎯 Best performance: {best_ratio*100:.0f}% data, {metric} = {best_performance:.4f}")
        
        # Save results
        results_path = os.path.join(self.output_dir, 'classification_results.json')
        with open(results_path, 'w') as f:
            # Exclude histories (too large) when saving
            save_results = {k: v for k, v in results.items() if k != 'histories'}
            json.dump(save_results, f, indent=2)
        print(f"💾 Results saved: {results_path}")
        
        return results

if __name__ == "__main__":
    # Test execution
    trainer = ClassificationTrainer(
        class0_dir='./manual_labeling_output/class0',
        class1_dir='./manual_labeling_output/class1',
        output_dir='./models/classification',
        batch_size=16,
        num_epochs=20,  # Reduced value for testing
        gpu_num=0
    )
    
    # Execute with reduced ratios for testing
    results = trainer.train_with_data_ratio(ratios=[0.1, 0.5, 1.0])
    print(f"Training completed! Results: {results}")