# modules/object_classifier.py
"""
객체 분류 모델 모듈
학습된 DenseNet 모델을 사용하여 탐지된 객체를 분류하는 모듈
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
    객체 분류기 클래스
    YOLO로 탐지된 객체를 Class 0 (Keep) 또는 Class 1 (Filter)로 분류
    """
    
    def __init__(self, model_path, device=None, conf_threshold=0.5, gpu_num=0):
        """
        객체 분류 모델 초기화
        
        Args:
            model_path (str): 사전 학습된 분류 모델 경로 (.pth 파일)
            device (torch.device): 연산 장치 (None이면 자동 선택)
            conf_threshold (float): 분류 신뢰도 임계값
            gpu_num (int): 사용할 GPU 번호
        """
        # 장치 설정
        if device is None:
            self.device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        
        print(f"🔧 분류 모델 장치: {self.device}")
        print(f"📁 모델 경로: {model_path}")
        print(f"🎯 신뢰도 임계값: {conf_threshold}")
        
        # 모델 구조 생성
        self.model = self._create_model()
        
        # 모델 가중치 로드
        self._load_model_weights()
        
        # 모델을 평가 모드로 설정
        self.model.eval()
        
        # 이미지 전처리 파이프라인 설정
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # DenseNet 입력 크기
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet 정규화
        ])
        
        print("✅ 객체 분류기 초기화 완료")
        
    def _create_model(self):
        """DenseNet121 모델 구조 생성"""
        print("🏗️ DenseNet121 모델 구조 생성 중...")
        
        # DenseNet121 모델 초기화 (사전 훈련 가중치 없이)
        model = models.densenet121(pretrained=False)
        
        # 이진 분류를 위해 마지막 레이어 수정
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),  # 드롭아웃 레이어
            nn.Linear(num_features, 2)  # 2 클래스 출력 (0: Keep, 1: Filter)
        )
        
        model = model.to(self.device)
        print("✅ 모델 구조 생성 완료")
        
        return model
        
    def _load_model_weights(self):
        """모델 가중치 로드"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        try:
            print("📥 모델 가중치 로딩 중...")
            
            # 가중치 로드
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            print("✅ 모델 가중치 로딩 성공")
            
        except RuntimeError as e:
            print(f"⚠️ 모델 가중치 로딩 실패: {e}")
            print("🔄 키 매핑을 시도합니다...")
            
            try:
                # DataParallel로 저장된 경우 'module.' 접두사 제거
                new_state_dict = {}
                for key, value in state_dict.items():
                    if 'module.' in key:
                        key = key.replace('module.', '')
                    new_state_dict[key] = value
                
                self.model.load_state_dict(new_state_dict)
                print("✅ 키 매핑 후 모델 가중치 로딩 성공")
                
            except Exception as e2:
                print(f"❌ 모델 로딩 최종 실패: {e2}")
                raise Exception(f"모델 로딩 실패: {e2}")
    
    def preprocess_image(self, image):
        """
        이미지 전처리
        
        Args:
            image (numpy.ndarray): OpenCV 형식의 이미지 (BGR)
            
        Returns:
            torch.Tensor: 전처리된 이미지 텐서
        """
        # 이미지 크기 검사
        if image.shape[0] < 10 or image.shape[1] < 10:
            print("⚠️ 이미지가 너무 작습니다 (10x10 미만)")
            return None
        
        # 이미지가 비어있는지 확인
        if image.size == 0:
            print("⚠️ 빈 이미지입니다")
            return None
        
        try:
            # OpenCV 이미지 (BGR)를 PIL 이미지 (RGB)로 변환
            if len(image.shape) == 3:
                # 컬러 이미지
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                # 그레이스케일 이미지
                pil_image = Image.fromarray(image).convert('RGB')
            
            # 전처리 적용
            input_tensor = self.transform(pil_image)
            
            # 배치 차원 추가 [C, H, W] -> [1, C, H, W]
            input_tensor = input_tensor.unsqueeze(0)
            
            return input_tensor.to(self.device)
            
        except Exception as e:
            print(f"⚠️ 이미지 전처리 실패: {str(e)}")
            return None
    
    def classify(self, image):
        """
        객체 이미지 분류 수행
        
        Args:
            image (numpy.ndarray): 분류할 객체 이미지 (OpenCV 형식, BGR)
            
        Returns:
            tuple: (예측 클래스, 신뢰도)
                - 예측 클래스: 0 (Keep) 또는 1 (Filter)
                - 신뢰도: 0.0 ~ 1.0 사이의 확률값
        """
        # 이미지 전처리
        input_tensor = self.preprocess_image(image)
        
        if input_tensor is None:
            # 전처리 실패 시 기본값 반환 (필터링)
            return 1, 0.0
        
        try:
            # 예측 수행 (그래디언트 계산 비활성화)
            with torch.no_grad():
                # 모델 forward pass
                outputs = self.model(input_tensor)
                
                # Softmax를 적용하여 확률 계산
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # 최대 확률과 해당 클래스 선택
                confidence, predicted_class = torch.max(probabilities, 1)
                
                # CPU로 이동하고 numpy 값으로 변환
                predicted_class = predicted_class.item()
                confidence = confidence.item()
                
                return predicted_class, confidence
                
        except Exception as e:
            print(f"⚠️ 분류 예측 실패: {str(e)}")
            # 오류 시 기본값 반환 (필터링)
            return 1, 0.0
    
    def classify_batch(self, images):
        """
        여러 이미지를 배치로 분류
        
        Args:
            images (list): OpenCV 이미지들의 리스트
            
        Returns:
            list: [(예측 클래스, 신뢰도), ...] 형태의 결과 리스트
        """
        if not images:
            return []
        
        print(f"🔍 배치 분류 시작: {len(images)}개 이미지")
        
        # 배치 텐서 준비
        batch_tensors = []
        valid_indices = []
        
        for i, image in enumerate(images):
            tensor = self.preprocess_image(image)
            if tensor is not None:
                batch_tensors.append(tensor.squeeze(0))  # 배치 차원 제거
                valid_indices.append(i)
        
        if not batch_tensors:
            print("⚠️ 유효한 이미지가 없습니다")
            return [(1, 0.0)] * len(images)
        
        try:
            # 배치 텐서 생성
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # 배치 예측
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidences, predicted_classes = torch.max(probabilities, 1)
                
                # CPU로 이동
                predicted_classes = predicted_classes.cpu().numpy()
                confidences = confidences.cpu().numpy()
            
            # 결과 매핑
            results = [(1, 0.0)] * len(images)  # 기본값으로 초기화
            
            for i, valid_idx in enumerate(valid_indices):
                results[valid_idx] = (int(predicted_classes[i]), float(confidences[i]))
            
            print(f"✅ 배치 분류 완료")
            return results
            
        except Exception as e:
            print(f"⚠️ 배치 분류 실패: {str(e)}")
            return [(1, 0.0)] * len(images)
    
    def get_class_name(self, class_id):
        """
        클래스 ID를 이름으로 변환
        
        Args:
            class_id (int): 클래스 ID (0 또는 1)
            
        Returns:
            str: 클래스 이름
        """
        class_names = {0: "Keep", 1: "Filter"}
        return class_names.get(class_id, "Unknown")
    
    def evaluate_confidence(self, confidence):
        """
        신뢰도 수준 평가
        
        Args:
            confidence (float): 신뢰도 값
            
        Returns:
            str: 신뢰도 수준 ("High", "Medium", "Low")
        """
        if confidence >= 0.8:
            return "High"
        elif confidence >= 0.6:
            return "Medium"
        else:
            return "Low"
    
    def classify_with_details(self, image):
        """
        상세 정보와 함께 분류 수행
        
        Args:
            image (numpy.ndarray): 분류할 객체 이미지
            
        Returns:
            dict: 상세 분류 결과
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
        탐지된 객체들을 분류하여 필터링
        
        Args:
            detected_objects (list): 탐지된 객체 정보 리스트
            images (list): 객체 이미지들의 리스트
            
        Returns:
            tuple: (유지할 객체들, 필터링된 객체들)
        """
        keep_objects = []
        filter_objects = []
        
        print(f"🔍 객체 필터링 시작: {len(detected_objects)}개 객체")
        
        # 배치 분류 수행
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
        
        print(f"📊 필터링 결과:")
        print(f"  - 유지: {len(keep_objects)}개")
        print(f"  - 필터링: {len(filter_objects)}개")
        
        return keep_objects, filter_objects
    
    def save_classification_results(self, results, save_path):
        """
        분류 결과를 파일로 저장
        
        Args:
            results (list): 분류 결과 리스트
            save_path (str): 저장 경로
        """
        try:
            import json
            
            # 결과를 JSON 직렬화 가능한 형태로 변환
            serializable_results = []
            for result in results:
                if isinstance(result, dict):
                    serializable_results.append(result)
                else:
                    # tuple인 경우 dict로 변환
                    serializable_results.append({
                        'predicted_class': result[0],
                        'confidence': result[1],
                        'class_name': self.get_class_name(result[0])
                    })
            
            with open(save_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"💾 분류 결과 저장: {save_path}")
            
        except Exception as e:
            print(f"⚠️ 결과 저장 실패: {str(e)}")
    
    def get_model_info(self):
        """
        모델 정보 반환
        
        Returns:
            dict: 모델 정보
        """
        # 모델 파라미터 수 계산
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
        """문자열 표현"""
        info = self.get_model_info()
        return f"""ObjectClassifier(
    Model: {info['model_architecture']}
    Device: {info['device']}
    Classes: {info['num_classes']} ({', '.join(info['class_names'])})
    Confidence Threshold: {info['confidence_threshold']}
    Parameters: {info['total_parameters']:,}
    Model Path: {info['model_path']}
)"""

# 테스트 및 유틸리티 함수들
def test_classifier(model_path, test_images_dir, output_dir=None):
    """
    분류기 테스트 함수
    
    Args:
        model_path (str): 분류 모델 경로
        test_images_dir (str): 테스트 이미지 디렉토리
        output_dir (str): 결과 저장 디렉토리
    """
    print("🧪 분류기 테스트 시작")
    
    # 분류기 초기화
    classifier = ObjectClassifier(model_path)
    print(classifier)
    
    # 테스트 이미지 로드
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
        print("❌ 테스트 이미지를 찾을 수 없습니다")
        return
    
    print(f"📸 {len(test_images)}개 테스트 이미지 로드됨")
    
    # 분류 수행
    results = []
    for i, (img, img_path) in enumerate(zip(test_images, image_paths)):
        result = classifier.classify_with_details(img)
        result['image_path'] = img_path
        result['filename'] = os.path.basename(img_path)
        results.append(result)
        
        print(f"📊 {i+1:3d}. {result['filename']:20s} -> "
              f"{result['class_name']:6s} ({result['confidence']:.3f}, {result['confidence_level']})")
    
    # 통계 출력
    keep_count = sum(1 for r in results if r['predicted_class'] == 0)
    filter_count = len(results) - keep_count
    avg_confidence = np.mean([r['confidence'] for r in results])
    
    print(f"\n📈 테스트 결과 통계:")
    print(f"  - Keep: {keep_count}개 ({keep_count/len(results)*100:.1f}%)")
    print(f"  - Filter: {filter_count}개 ({filter_count/len(results)*100:.1f}%)")
    print(f"  - 평균 신뢰도: {avg_confidence:.3f}")
    
    # 결과 저장
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        classifier.save_classification_results(results, 
                                             os.path.join(output_dir, 'test_results.json'))

if __name__ == "__main__":
    # 테스트 실행 예시
    print("🔍 ObjectClassifier 모듈 테스트")
    
    # 모델 경로 설정 (실제 경로로 변경 필요)
    model_path = "./models/classification/densenet121_100.pth"
    test_images_dir = "./test_images"
    output_dir = "./test_results"
    
    if os.path.exists(model_path):
        test_classifier(model_path, test_images_dir, output_dir)
    else:
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        print("분류 모델을 먼저 학습시켜주세요.")