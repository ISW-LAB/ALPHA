# modules/classification_trainer.py
"""
Classification 모델 학습 모듈
수동 라벨링된 객체 데이터로 DenseNet 분류 모델을 학습시키는 모듈
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
    """객체 이미지 데이터셋 클래스"""
    
    def __init__(self, class0_dir, class1_dir, transform=None):
        """
        객체 이미지 데이터셋 초기화
        
        Args:
            class0_dir (str): 클래스 0 이미지 디렉토리 경로
            class1_dir (str): 클래스 1 이미지 디렉토리 경로
            transform: 이미지 변환기
        """
        self.transform = transform
        self.class_dirs = [class0_dir, class1_dir]
        self.samples = []
        
        # 각 클래스 디렉토리에서 이미지 로드
        for class_idx, class_dir in enumerate(self.class_dirs):
            if not os.path.isdir(class_dir):
                print(f"⚠️ 경고: {class_dir} 디렉토리가 존재하지 않습니다.")
                continue
                
            image_count = 0
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    try:
                        # 이미지 파일이 올바른지 확인
                        with Image.open(img_path) as img:
                            img.verify()
                        self.samples.append((img_path, class_idx))
                        image_count += 1
                    except Exception as e:
                        print(f"⚠️ 손상된 이미지 파일 건너뜀: {img_path} - {str(e)}")
            
            print(f"클래스 {class_idx}: {image_count}개 이미지 로드됨 ({class_dir})")
        
        print(f"📊 총 {len(self.samples)}개의 이미지가 로드되었습니다.")
        
        if len(self.samples) == 0:
            raise ValueError("로드된 이미지가 없습니다. 디렉토리 경로를 확인하세요.")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        데이터셋에서 항목 가져오기
        
        Args:
            idx (int): 인덱스
            
        Returns:
            tuple: (이미지, 클래스 라벨)
        """
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ 이미지 로드 실패: {img_path} - {str(e)}")
            # 오류 시 빈 이미지 반환
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class ClassificationTrainer:
    """Classification 모델 학습기"""
    
    def __init__(self, class0_dir, class1_dir, output_dir, batch_size=16, 
                 num_epochs=30, gpu_num=0, random_seed=13):
        """
        Classification 학습기 초기화
        
        Args:
            class0_dir (str): 클래스 0 이미지 디렉토리
            class1_dir (str): 클래스 1 이미지 디렉토리
            output_dir (str): 결과 저장 디렉토리
            batch_size (int): 배치 크기
            num_epochs (int): 학습 에폭 수
            gpu_num (int): 사용할 GPU 번호
            random_seed (int): 랜덤 시드
        """
        self.class0_dir = class0_dir
        self.class1_dir = class1_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.gpu_num = gpu_num
        self.random_seed = random_seed
        
        # 재현성을 위한 시드 설정
        self.set_seed()
        
        # 디바이스 설정
        self.device = torch.device(f"cuda:{self.gpu_num}" if torch.cuda.is_available() else "cpu")
        print(f"🔧 사용 장치: {self.device}")
        
        # 결과 저장 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 이미지 변환기 정의
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),  # 데이터 증강
                transforms.RandomRotation(degrees=10),    # 회전 증강
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 증강
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        print("✅ Classification Trainer 초기화 완료")
    
    def set_seed(self):
        """재현성을 위한 시드 설정"""
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def init_weights(self, m):
        """가중치 초기화 함수"""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def create_model(self):
        """DenseNet121 모델 생성"""
        print("🏗️ DenseNet121 모델 생성 중...")
        
        # DenseNet121 모델 초기화 (사전 훈련된 가중치 사용)
        model = models.densenet121(pretrained=True)
        
        # 특징 추출 레이어 대부분 고정 (과적합 방지)
        for param in model.features.parameters():
            param.requires_grad = False
            
        # 마지막 dense block만 학습 가능하게 설정
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
            
        # 마지막 정규화 레이어도 학습 가능하게 설정
        for param in model.features.norm5.parameters():
            param.requires_grad = True
        
        # 분류기 레이어 교체 (드롭아웃 포함)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),  # 과적합 방지
            nn.Linear(num_features, 2)  # 이진 분류
        )
        
        # 새로 추가한 레이어 가중치 초기화
        model.classifier.apply(self.init_weights)
        
        model = model.to(self.device)
        print("✅ 모델 생성 및 GPU 로딩 완료")
        
        return model
    
    def train_model(self, model, dataloaders, criterion, optimizer, scheduler, ratio, 
                   metric='val_f1', patience=10):
        """
        모델 학습 함수
        
        Args:
            model: 학습할 모델
            dataloaders: 데이터 로더 딕셔너리
            criterion: 손실 함수
            optimizer: 옵티마이저
            scheduler: 학습률 스케줄러
            ratio: 현재 사용 중인 데이터 비율
            metric: 모델 선택 기준 메트릭
            patience: 조기 종료 인내심
            
        Returns:
            tuple: (학습 기록, 최고 성능, 최고 성능 에폭)
        """
        print(f"🚀 모델 학습 시작 (데이터 비율: {ratio*100:.0f}%)")
        
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
        
        print(f"📊 학습 설정:")
        print(f"  - 에폭: {self.num_epochs}")
        print(f"  - 조기 종료 기준: {metric}")
        print(f"  - 조기 종료 인내심: {patience}")
        
        for epoch in range(self.num_epochs):
            print(f'\n📅 Epoch {epoch+1}/{self.num_epochs}')
            print('-' * 50)
            
            # 각 에폭은 학습과 검증 단계를 가짐
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # 학습 모드
                else:
                    model.eval()   # 평가 모드
                    
                running_loss = 0.0
                all_preds = []
                all_labels = []
                
                # 진행률 표시를 위한 tqdm
                pbar = tqdm(dataloaders[phase], desc=f'{phase.upper()}')
                
                # 데이터 반복
                for inputs, labels in pbar:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # 매개변수 그래디언트 초기화
                    optimizer.zero_grad()
                    
                    # 순전파
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        # 학습 단계에서만 역전파 + 최적화
                        if phase == 'train':
                            loss.backward()
                            # 그래디언트 클리핑 (과적합 방지)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            
                    # 통계 수집
                    running_loss += loss.item() * inputs.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    # 진행률 표시 업데이트
                    pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                
                # 에폭 성능 계산
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = accuracy_score(all_labels, all_preds)
                
                print(f'{phase.upper()} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
                
                # 검증 단계에서 추가 메트릭 계산
                if phase == 'val':
                    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
                    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
                    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
                    
                    print(f'VAL - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
                    
                    # 기록 저장
                    history['val_loss'].append(epoch_loss)
                    history['val_acc'].append(epoch_acc)
                    history['val_precision'].append(precision)
                    history['val_recall'].append(recall)
                    history['val_f1'].append(f1)
                    
                    # 학습률 스케줄러 업데이트
                    if scheduler is not None:
                        scheduler.step(epoch_loss)
                        current_lr = optimizer.param_groups[0]['lr']
                        print(f'📉 현재 학습률: {current_lr:.6f}')
                    
                    # 최고 성능 모델 체크
                    if metric == 'val_acc':
                        current_score = epoch_acc
                    elif metric == 'val_precision':
                        current_score = precision
                    elif metric == 'val_recall':
                        current_score = recall
                    elif metric == 'val_f1':
                        current_score = f1
                    else:
                        current_score = f1  # 기본값
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_epoch = epoch
                        best_model_wts = model.state_dict().copy()
                        print(f"🎉 새로운 최고 성능! (에폭 {epoch+1}, {metric} = {current_score:.4f})")
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                        print(f"⏳ 성능 향상 없음: {early_stop_counter}/{patience}")
                        
                        if early_stop_counter >= patience:
                            print(f"🛑 조기 종료: {patience}번 연속 성능 향상 없음")
                            break
                else:
                    # 훈련 단계 기록
                    history['train_loss'].append(epoch_loss)
                    history['train_acc'].append(epoch_acc)
            
            # 조기 종료 확인
            if early_stop_counter >= patience:
                print(f"🏁 학습을 {epoch+1}번째 에폭에서 조기 종료합니다.")
                break
        
        # 학습 완료 정보
        time_elapsed = time.time() - start_time
        print(f'\n⏰ 학습 완료: {time_elapsed // 60:.0f}분 {time_elapsed % 60:.0f}초')
        print(f'🏆 최고 성능: 에폭 {best_epoch+1}, {metric} = {best_score:.4f}')
        
        # 최고 성능 모델 저장
        model_save_path = os.path.join(self.output_dir, f'densenet121_{int(ratio*100)}.pth')
        torch.save(best_model_wts, model_save_path)
        print(f"💾 최고 성능 모델 저장: {model_save_path}")
        
        # 학습 기록 저장
        history_save_path = os.path.join(self.output_dir, f'training_history_{int(ratio*100)}.json')
        with open(history_save_path, 'w') as f:
            # numpy 배열을 리스트로 변환하여 JSON 직렬화 가능하게 만듦
            json_history = {k: [float(x) for x in v] for k, v in history.items()}
            json.dump(json_history, f, indent=2)
        print(f"📊 학습 기록 저장: {history_save_path}")
        
        # 모델을 최고 성능 상태로 복원
        model.load_state_dict(best_model_wts)
        
        return history, best_score, best_epoch+1
    
    def plot_training_history(self, history, ratio):
        """학습 기록 시각화"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training History - {int(ratio*100)}% Data', fontsize=16)
            
            # Loss 그래프
            axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
            axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Accuracy 그래프
            axes[0, 1].plot(history['train_acc'], label='Train Acc', color='blue')
            axes[0, 1].plot(history['val_acc'], label='Val Acc', color='red')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Precision, Recall, F1 그래프
            axes[1, 0].plot(history['val_precision'], label='Precision', color='green')
            axes[1, 0].plot(history['val_recall'], label='Recall', color='orange')
            axes[1, 0].plot(history['val_f1'], label='F1-Score', color='purple')
            axes[1, 0].set_title('Validation Metrics')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # 최종 성능 표시
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
            
            print(f"📈 학습 그래프 저장: {plot_path}")
            
        except Exception as e:
            print(f"⚠️ 그래프 저장 실패: {str(e)}")
    
    def train_with_data_ratio(self, ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
                             metric='val_f1'):
        """
        데이터 비율을 조절하며 모델 학습 수행
        
        Args:
            ratios: 사용할 데이터 비율 리스트
            metric: 모델 선택 기준 메트릭
            
        Returns:
            dict: 학습 결과 딕셔너리
        """
        print("="*80)
        print("🎯 Classification 모델 학습 시작")
        print("="*80)
        
        # 전체 데이터셋 로드
        print("📂 데이터셋 로딩 중...")
        try:
            full_dataset = ObjectDataset(self.class0_dir, self.class1_dir, 
                                       transform=self.data_transforms['train'])
        except Exception as e:
            print(f"❌ 데이터셋 로딩 실패: {str(e)}")
            return {}
        
        # 클래스별 인덱스 추출
        class_indices = {0: [], 1: []}
        for i, (_, label) in enumerate(full_dataset.samples):
            class_indices[label].append(i)
        
        print(f"📊 클래스별 데이터 분포:")
        print(f"  - 클래스 0: {len(class_indices[0])}개 이미지")
        print(f"  - 클래스 1: {len(class_indices[1])}개 이미지")
        
        # 데이터가 너무 적은 경우 경고
        total_samples = len(class_indices[0]) + len(class_indices[1])
        if total_samples < 100:
            print("⚠️ 경고: 데이터 수가 매우 적습니다(100개 미만). 추가 데이터 수집을 권장합니다.")
        
        val_ratio = 0.15  # 검증 세트 비율
        results = {'ratios': [], 'best_scores': [], 'best_epochs': [], 'histories': []}
        
        print(f"\n🔄 학습할 데이터 비율: {ratios}")
        print(f"🎯 모델 선택 기준: {metric}")
        
        # 각 데이터 비율에 대해 학습 실행
        for i, ratio in enumerate(ratios):
            print(f"\n{'='*60}")
            print(f"📊 데이터 비율: {ratio*100:.0f}% ({i+1}/{len(ratios)})")
            print(f"{'='*60}")
            
            try:
                # 각 클래스에서 비율만큼 데이터 선택
                train_indices = []
                for label in [0, 1]:
                    class_size = len(class_indices[label])
                    num_samples = int(class_size * ratio)
                    
                    if num_samples < 3:
                        print(f"⚠️ 경고: 클래스 {label}의 데이터가 너무 적습니다 ({num_samples}개)")
                    
                    # 점층적 선택 (항상 동일한 순서)
                    train_indices.extend(class_indices[label][:num_samples])
                
                # 학습/검증 분할
                val_size = max(int(len(train_indices) * val_ratio), 2)  # 최소 2개는 검증용
                train_size = len(train_indices) - val_size
                
                # 인덱스 섞기 (재현 가능하도록)
                random.shuffle(train_indices)
                train_subset = Subset(full_dataset, train_indices[val_size:])
                
                # 검증 데이터셋 (별도 변환 적용)
                val_dataset = ObjectDataset(self.class0_dir, self.class1_dir, 
                                          transform=self.data_transforms['val'])
                val_subset = Subset(val_dataset, train_indices[:val_size])
                
                print(f"📊 데이터 분할:")
                print(f"  - 학습 세트: {len(train_subset)}개")
                print(f"  - 검증 세트: {len(val_subset)}개")
                
                # 배치 크기 조정 (데이터가 적은 경우)
                actual_batch_size = min(self.batch_size, len(train_subset) // 2)
                actual_batch_size = max(actual_batch_size, 1)  # 최소 1개
                
                # 데이터 로더 생성
                dataloaders = {
                    'train': DataLoader(train_subset, batch_size=actual_batch_size, 
                                      shuffle=True, num_workers=4, pin_memory=True),
                    'val': DataLoader(val_subset, batch_size=actual_batch_size, 
                                    shuffle=False, num_workers=4, pin_memory=True)
                }
                
                print(f"⚙️ 실제 배치 크기: {actual_batch_size}")
                
                # 모델 생성
                model = self.create_model()
                
                # 클래스 불균형 처리를 위한 가중치 계산
                class_samples = [len(class_indices[0]), len(class_indices[1])]
                if min(class_samples) > 0:
                    weights = [len(full_dataset) / (2 * count) if count > 0 else 1.0 
                             for count in class_samples]
                    class_weights = torch.FloatTensor(weights).to(self.device)
                    criterion = nn.CrossEntropyLoss(weight=class_weights)
                    print(f"⚖️ 클래스 가중치 적용: {[f'{w:.3f}' for w in weights]}")
                else:
                    criterion = nn.CrossEntropyLoss()
                
                # 옵티마이저 설정
                trainable_params = [param for param in model.parameters() if param.requires_grad]
                optimizer = optim.Adam(
                    trainable_params, 
                    lr=0.0005,  # 낮은 학습률
                    weight_decay=0.0001  # L2 정규화
                )
                
                # 학습률 스케줄러
                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,
                    patience=5,
                    verbose=True,
                    min_lr=1e-7
                )
                
                # 모델 학습
                history, best_score, best_epoch = self.train_model(
                    model, dataloaders, criterion, optimizer, scheduler, 
                    ratio, metric, patience=10
                )
                
                # 결과 저장
                results['ratios'].append(ratio)
                results['best_scores'].append(best_score)
                results['best_epochs'].append(best_epoch)
                results['histories'].append(history)
                
                # 학습 그래프 저장
                self.plot_training_history(history, ratio)
                
                print(f"✅ {ratio*100:.0f}% 데이터 학습 완료:")
                print(f"   - 최고 성능 ({metric}): {best_score:.4f} (에폭 {best_epoch})")
                
            except Exception as e:
                print(f"❌ {ratio*100:.0f}% 데이터 학습 실패: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # 실패한 경우에도 결과에 기록 (0으로)
                results['ratios'].append(ratio)
                results['best_scores'].append(0.0)
                results['best_epochs'].append(0)
                results['histories'].append({})
        
        # 전체 결과 요약
        print("\n" + "="*80)
        print("🏆 Classification 학습 결과 요약")
        print("="*80)
        
        for i, ratio in enumerate(results['ratios']):
            if i < len(results['best_scores']):
                score = results['best_scores'][i]
                epoch = results['best_epochs'][i]
                status = "✅" if score > 0 else "❌"
                print(f"{status} 데이터 비율 {ratio*100:3.0f}%: {metric} = {score:.4f} (에폭 {epoch})")
        
        # 최고 성능 모델 정보
        if results['best_scores']:
            best_idx = np.argmax(results['best_scores'])
            best_ratio = results['ratios'][best_idx]
            best_performance = results['best_scores'][best_idx]
            print(f"\n🎯 최고 성능: {best_ratio*100:.0f}% 데이터, {metric} = {best_performance:.4f}")
        
        # 결과 저장
        results_path = os.path.join(self.output_dir, 'classification_results.json')
        with open(results_path, 'w') as f:
            # 히스토리는 너무 크므로 제외하고 저장
            save_results = {k: v for k, v in results.items() if k != 'histories'}
            json.dump(save_results, f, indent=2)
        print(f"💾 결과 저장: {results_path}")
        
        return results

if __name__ == "__main__":
    # 테스트 실행
    trainer = ClassificationTrainer(
        class0_dir='./manual_labeling_output/class0',
        class1_dir='./manual_labeling_output/class1',
        output_dir='./models/classification',
        batch_size=16,
        num_epochs=20,  # 테스트용으로 낮춘 값
        gpu_num=0
    )
    
    # 테스트용으로 적은 비율만 실행
    results = trainer.train_with_data_ratio(ratios=[0.1, 0.5, 1.0])
    print(f"학습 완료! 결과: {results}")