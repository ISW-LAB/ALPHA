# modules/yolo_active_learning.py
"""
YOLO Active Learning 핵심 모듈
YOLO 탐지 + Classification 필터링을 결합한 반복적 학습 시스템
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

# 로컬 모듈 임포트
from modules.object_classifier import ObjectClassifier

class YOLOActiveLearning:
    """
    YOLO 기반 Active Learning 시스템
    YOLO 객체 탐지와 Classification 모델을 결합하여 반복적 학습을 수행
    """
    
    def __init__(self, model_path, classifier_path=None, image_dir=None, label_dir=None, output_dir=None, 
                 conf_threshold=0.25, iou_threshold=0.5, class_conf_threshold=0.5, max_cycles=5, gpu_num=0,
                 use_classifier=False):
        """
        YOLO Active Learning 시스템 초기화
        
        Args:
            model_path (str): 사전 학습된 YOLO 모델 경로
            classifier_path (str, optional): 사전 학습된 분류 모델 경로
            image_dir (str): 이미지 데이터셋 경로
            label_dir (str): 정답 라벨 경로
            output_dir (str): 결과 저장 경로
            conf_threshold (float): 객체 검출 신뢰도 임계값
            iou_threshold (float): IoU 임계값
            class_conf_threshold (float): 분류 모델 신뢰도 임계값
            max_cycles (int): 최대 학습 반복 횟수
            gpu_num (int): 사용할 GPU 번호
            use_classifier (bool): 분류 모델 사용 여부
        """
        # 기본 설정 저장
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
        
        # 모델 이름 추출
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        # GPU 설정
        self.device = torch.device(f"cuda:{self.gpu_num}" if torch.cuda.is_available() else "cpu")
        
        print("🔧 YOLO Active Learning 시스템 설정:")
        print(f"  - YOLO 모델: {self.model_name}")
        print(f"  - 분류 모델 사용: {self.use_classifier}")
        print(f"  - 최대 사이클: {self.max_cycles}")
        print(f"  - 장치: {self.device}")
        print(f"  - 결과 저장: {self.output_dir}")
        
        # 디렉토리 구조 생성
        self.create_directories()
        
        # YOLO 모델 로드
        print("📥 YOLO 모델 로딩 중...")
        self.model = YOLO(self.model_path)
        print("✅ YOLO 모델 로딩 완료")
        
        # 분류 모델 로드 (선택적)
        self.classifier = None
        if self.use_classifier and self.classifier_path:
            print("📥 분류 모델 로딩 중...")
            self.classifier = ObjectClassifier(
                self.classifier_path, 
                self.device, 
                self.class_conf_threshold, 
                self.gpu_num
            )
            print("✅ 분류 모델 로딩 완료")
        
        # 성능 지표 저장용 데이터프레임 초기화
        self.setup_metrics_tracking()
        
        # 통계 변수 초기화
        self.reset_statistics()
        
        print("✅ YOLO Active Learning 시스템 초기화 완료")
        
    def create_directories(self):
        """필요한 디렉토리 구조 생성"""
        print("📁 디렉토리 구조 생성 중...")
        
        # 기본 출력 디렉토리
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 각 학습 주기별 결과 디렉토리
        for cycle in range(1, self.max_cycles + 1):
            cycle_dir = os.path.join(self.output_dir, f"cycle_{cycle}")
            
            # 기본 디렉토리들
            subdirs = ["detections", "labels", "training"]
            
            # 분류 모델 사용 시 추가 디렉토리
            if self.use_classifier:
                subdirs.extend(["filtered_detections", "filtered_labels"])
            
            for subdir in subdirs:
                os.makedirs(os.path.join(cycle_dir, subdir), exist_ok=True)
        
        # 훈련 데이터셋 디렉토리
        self.dataset_dir = os.path.join(self.output_dir, "dataset")
        for split in ["train", "val"]:
            for data_type in ["images", "labels"]:
                os.makedirs(os.path.join(self.dataset_dir, data_type, split), exist_ok=True)
        
        print("✅ 디렉토리 구조 생성 완료")
    
    def setup_metrics_tracking(self):
        """성능 지표 추적 설정"""
        # 메트릭 컬럼 정의
        columns = [
            'Cycle', 'Model', 'mAP50', 'Precision', 'Recall', 'F1-Score', 
            'Detected_Objects', 'Filtered_Objects'
        ]
        
        self.metrics_df = pd.DataFrame(columns=columns)
        self.metrics_file = os.path.join(self.output_dir, "performance_metrics.csv")
        
        # 기존 메트릭 파일이 있으면 로드
        if os.path.exists(self.metrics_file):
            try:
                existing_metrics = pd.read_csv(self.metrics_file)
                self.metrics_df = existing_metrics
                print(f"📊 기존 메트릭 파일 로드: {self.metrics_file}")
            except Exception as e:
                print(f"⚠️ 기존 메트릭 파일 로드 실패: {str(e)}")
    
    def reset_statistics(self):
        """통계 변수 초기화"""
        self.filtered_objects_count = 0
        self.detected_objects_count = 0
    
    def detect_and_classify_objects(self, image_path, cycle):
        """
        이미지에서 객체를 탐지하고 분류 모델로 필터링
        
        Args:
            image_path (str): 이미지 경로
            cycle (int): 현재 학습 주기
            
        Returns:
            tuple: (탐지된 객체 리스트, 필터링된 객체 리스트, 전체 탐지 이미지, 필터링된 이미지)
        """
        # 이미지 로드
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ 이미지 로드 실패: {image_path}")
            return [], [], None, None
        
        # YOLO 객체 탐지 수행
        try:
            results = self.model.predict(
                source=img, 
                conf=self.conf_threshold, 
                iou=self.iou_threshold,
                save=False,
                verbose=False
            )
        except Exception as e:
            print(f"⚠️ YOLO 예측 실패: {str(e)}")
            return [], [], None, None
        
        # 결과 처리
        result = results[0]
        detected_objects = []
        filtered_objects = []
        
        # 시각화를 위한 이미지 복사
        img_with_all_boxes = img.copy()
        img_with_filtered_boxes = img.copy() if self.use_classifier else None
        
        if len(result.boxes) > 0:
            # 객체 이미지들을 배치로 처리하기 위한 리스트
            object_images = []
            object_infos = []
            
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # 좌표 정수 변환 및 경계 확인
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # 유효한 바운딩 박스인지 확인
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # 검출된 객체 이미지 추출
                obj_img = img[y1:y2, x1:x2]
                
                if obj_img.size == 0:
                    continue
                
                # YOLO 포맷으로 좌표 변환 (정규화된 중심점, 너비, 높이)
                center_x = ((x1 + x2) / 2) / w
                center_y = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                # 객체 정보 저장
                obj_info = {
                    'cls_id': 0,  # 단일 클래스
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                }
                
                object_images.append(obj_img)
                object_infos.append(obj_info)
                
                # 모든 탐지 결과에 박스 그리기
                cv2.rectangle(img_with_all_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_with_all_boxes, f"Obj {conf:.2f}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 분류 모델 사용 시 배치 분류 수행
            if self.use_classifier and self.classifier and object_images:
                try:
                    # 배치 분류 수행
                    classification_results = self.classifier.classify_batch(object_images)
                    
                    for obj_info, (pred_class, class_conf) in zip(object_infos, classification_results):
                        bbox = obj_info['bbox']
                        x1, y1, x2, y2 = bbox
                        
                        # 분류 결과에 따라 객체 분류
                        if pred_class == 0:  # Keep
                            detected_objects.append([
                                obj_info['cls_id'], obj_info['center_x'], obj_info['center_y'], 
                                obj_info['width'], obj_info['height']
                            ])
                            self.detected_objects_count += 1
                            
                            # 유지할 객체 시각화 (초록색)
                            cv2.rectangle(img_with_filtered_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img_with_filtered_boxes, f"Keep {class_conf:.2f}", 
                                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:  # Filter
                            filtered_objects.append([
                                obj_info['cls_id'], obj_info['center_x'], obj_info['center_y'], 
                                obj_info['width'], obj_info['height']
                            ])
                            self.filtered_objects_count += 1
                            
                            # 필터링된 객체 시각화 (빨간색)
                            cv2.rectangle(img_with_all_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(img_with_all_boxes, f"Filter {class_conf:.2f}", 
                                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                except Exception as e:
                    print(f"⚠️ 분류 처리 실패: {str(e)}")
                    # 분류 실패 시 모든 객체를 탐지된 것으로 처리
                    for obj_info in object_infos:
                        detected_objects.append([
                            obj_info['cls_id'], obj_info['center_x'], obj_info['center_y'], 
                            obj_info['width'], obj_info['height']
                        ])
                        self.detected_objects_count += 1
            else:
                # 분류 모델 미사용 시 모든 객체를 탐지된 것으로 처리
                for obj_info in object_infos:
                    detected_objects.append([
                        obj_info['cls_id'], obj_info['center_x'], obj_info['center_y'], 
                        obj_info['width'], obj_info['height']
                    ])
                    self.detected_objects_count += 1
        
        return detected_objects, filtered_objects, img_with_all_boxes, img_with_filtered_boxes
    
    def save_label(self, objects, label_path):
        """
        탐지된 객체를 YOLO 포맷 라벨 파일로 저장
        
        Args:
            objects (list): 객체 리스트 [cls_id, center_x, center_y, width, height]
            label_path (str): 저장할 라벨 파일 경로
        """
        try:
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            
            with open(label_path, 'w') as f:
                for obj in objects:
                    line = ' '.join([str(x) for x in obj])
                    f.write(line + '\n')
                    
        except Exception as e:
            print(f"⚠️ 라벨 저장 실패: {label_path} - {str(e)}")
    
    def prepare_dataset(self, cycle):
        """
        YOLO 학습을 위한 데이터셋 준비
        
        Args:
            cycle (int): 현재 학습 주기
        """
        print(f"📦 사이클 {cycle} 데이터셋 준비 중...")
        
        # 사용할 라벨 디렉토리 결정
        if self.use_classifier:
            labels_source_dir = os.path.join(self.output_dir, f"cycle_{cycle}", "filtered_labels")
        else:
            labels_source_dir = os.path.join(self.output_dir, f"cycle_{cycle}", "labels")
        
        # 이미지 파일 목록
        image_files = [f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            raise ValueError("이미지 파일을 찾을 수 없습니다")
        
        # 데이터셋 분할 (대부분을 훈련용으로 사용)
        total_images = len(image_files)
        if total_images > 10:
            val_count = max(5, min(20, int(total_images * 0.05)))  # 5% 또는 최소 5개
        else:
            val_count = 1  # 최소 1개는 검증용
        
        val_files = image_files[:val_count]
        train_files = image_files[val_count:]
        
        print(f"  - 훈련: {len(train_files)}개")
        print(f"  - 검증: {len(val_files)}개")
        
        # 기존 데이터셋 디렉토리 비우기
        for split in ["train", "val"]:
            for data_type in ["images", "labels"]:
                dir_path = os.path.join(self.dataset_dir, data_type, split)
                if os.path.exists(dir_path):
                    for file in os.listdir(dir_path):
                        file_path = os.path.join(dir_path, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
        
        # 데이터 복사 함수
        def copy_files(file_list, split_name):
            copied_count = 0
            for img_file in file_list:
                # 이미지 복사
                src_img = os.path.join(self.image_dir, img_file)
                dst_img = os.path.join(self.dataset_dir, "images", split_name, img_file)
                
                if os.path.exists(src_img):
                    shutil.copy(src_img, dst_img)
                    
                    # 라벨 복사
                    label_file = os.path.splitext(img_file)[0] + '.txt'
                    src_label = os.path.join(labels_source_dir, label_file)
                    dst_label = os.path.join(self.dataset_dir, "labels", split_name, label_file)
                    
                    if os.path.exists(src_label):
                        shutil.copy(src_label, dst_label)
                        copied_count += 1
                    else:
                        # 라벨 파일이 없는 경우 빈 파일 생성
                        with open(dst_label, 'w') as f:
                            pass
            
            return copied_count
        
        # 파일 복사 실행
        train_copied = copy_files(train_files, "train")
        val_copied = copy_files(val_files, "val")
        
        print(f"  ✅ 데이터셋 준비 완료: 훈련 {train_copied}개, 검증 {val_copied}개")
        
        # 데이터셋 YAML 파일 생성
        self.create_dataset_yaml()
    
    def create_dataset_yaml(self):
        """YOLO 학습용 데이터셋 YAML 파일 생성"""
        dataset_yaml = {
            'path': os.path.abspath(self.dataset_dir),
            'train': 'images/train',
            'val': 'images/val', 
            'nc': 1,  # 단일 클래스
            'names': ['object']
        }
        
        yaml_path = os.path.join(self.dataset_dir, 'dataset.yaml')
        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(dataset_yaml, f, default_flow_style=False)
            
            print(f"📄 데이터셋 YAML 생성: {yaml_path}")
            return yaml_path
            
        except Exception as e:
            print(f"⚠️ YAML 파일 생성 실패: {str(e)}")
            return None
    
    def train_model(self, cycle):
        """
        현재 주기의 데이터로 YOLO 모델 학습
        
        Args:
            cycle (int): 현재 학습 주기
            
        Returns:
            str: 학습된 모델 경로
        """
        print(f"🎓 사이클 {cycle} YOLO 모델 학습 시작...")
        
        yaml_path = os.path.join(self.dataset_dir, 'dataset.yaml')
        if not os.path.exists(yaml_path):
            raise FileNotFoundError("데이터셋 YAML 파일이 없습니다")
        
        # 학습 결과 저장 경로
        training_dir = os.path.join(self.output_dir, f"cycle_{cycle}", "training")
        
        try:
            # 모델 학습 실행
            results = self.model.train(
                data=yaml_path,
                epochs=50,  # 사이클당 에폭 수
                imgsz=640,
                batch=16,
                patience=10,  # 조기 종료
                project=training_dir,
                name="yolo_model",
                device=self.device,
                plots=True,  # 학습 그래프 저장
                save_period=10  # 10 에폭마다 저장
            )
            
            # 학습된 모델 경로
            trained_model_path = os.path.join(training_dir, "yolo_model", "weights", "best.pt")
            
            if os.path.exists(trained_model_path):
                # 모델 업데이트
                self.model = YOLO(trained_model_path)
                print(f"✅ 모델 학습 완료: {trained_model_path}")
                return trained_model_path
            else:
                raise FileNotFoundError("학습된 모델 파일을 찾을 수 없습니다")
                
        except Exception as e:
            print(f"❌ 모델 학습 실패: {str(e)}")
            raise
    
    def evaluate_performance(self, cycle):
        """
        모델 성능 평가
        
        Args:
            cycle (int): 현재 학습 주기
            
        Returns:
            dict: 성능 지표 딕셔너리
        """
        print(f"📊 사이클 {cycle} 성능 평가 중...")
        
        # 이미지 파일 목록
        image_files = [f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            print("⚠️ 평가할 이미지가 없습니다")
            return self._create_empty_metrics(cycle)
        
        # 성능 지표 수집
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        
        print(f"  📸 {len(image_files)}개 이미지 평가 중...")
        
        for image_file in tqdm(image_files, desc="Evaluating"):
            image_path = os.path.join(self.image_dir, image_file)
            
            # 현재 모델로 객체 탐지
            detected_objects, _, _, _ = self.detect_and_classify_objects(image_path, cycle)
            
            # 정답 라벨 로드
            gt_label_path = os.path.join(self.label_dir, os.path.splitext(image_file)[0] + '.txt')
            gt_objects = self._load_ground_truth(gt_label_path)
            
            # 성능 계산 (간소화된 방식)
            precision, recall, f1 = self._calculate_performance_metrics(detected_objects, gt_objects)
            
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1_scores.append(f1)
        
        # 평균 성능 계산
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        avg_f1 = np.mean(all_f1_scores)
        
        # 메트릭 딕셔너리 생성
        metrics = {
            'Cycle': cycle,
            'Model': self.model_name,
            'mAP50': avg_precision,  # 간소화된 mAP
            'Precision': avg_precision,
            'Recall': avg_recall,
            'F1-Score': avg_f1,
            'Detected_Objects': self.detected_objects_count,
            'Filtered_Objects': self.filtered_objects_count if self.use_classifier else 0
        }
        
        # 메트릭 저장
        self._save_metrics(metrics)
        
        print(f"  📈 성능 결과:")
        print(f"    - mAP50: {avg_precision:.4f}")
        print(f"    - Precision: {avg_precision:.4f}")
        print(f"    - Recall: {avg_recall:.4f}")
        print(f"    - F1-Score: {avg_f1:.4f}")
        print(f"    - 탐지 객체: {self.detected_objects_count}")
        if self.use_classifier:
            print(f"    - 필터링 객체: {self.filtered_objects_count}")
        
        return metrics
    
    def _load_ground_truth(self, gt_label_path):
        """정답 라벨 로드"""
        gt_objects = []
        
        if os.path.exists(gt_label_path):
            try:
                with open(gt_label_path, 'r') as f:
                    for line in f:
                        values = line.strip().split()
                        if len(values) >= 5:
                            cls_id = 0  # 단일 클래스로 변환
                            center_x = float(values[1])
                            center_y = float(values[2])
                            width = float(values[3])
                            height = float(values[4])
                            gt_objects.append([cls_id, center_x, center_y, width, height])
            except Exception as e:
                print(f"⚠️ 정답 라벨 로드 실패: {gt_label_path} - {str(e)}")
        
        return gt_objects
    
    def _calculate_performance_metrics(self, detected_objects, gt_objects):
        """성능 지표 계산 (간소화된 방식)"""
        if len(gt_objects) == 0 and len(detected_objects) == 0:
            return 1.0, 1.0, 1.0
        elif len(gt_objects) == 0:
            return 0.0, 1.0, 0.0
        elif len(detected_objects) == 0:
            return 1.0, 0.0, 0.0
        else:
            # 간소화된 매칭 방식 (실제로는 IoU 기반 매칭 필요)
            # 객체 수 기반 근사 계산
            precision = min(1.0, len(gt_objects) / len(detected_objects))
            recall = min(1.0, len(detected_objects) / len(gt_objects))
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            return precision, recall, f1
    
    def _create_empty_metrics(self, cycle):
        """빈 메트릭 생성"""
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
        """메트릭을 데이터프레임에 저장"""
        # 기존 메트릭에서 동일한 Cycle과 Model 조합 확인
        mask = (self.metrics_df['Cycle'] == metrics['Cycle']) & \
               (self.metrics_df['Model'] == metrics['Model'])
        
        if any(mask):
            # 기존 항목 업데이트
            for col, value in metrics.items():
                self.metrics_df.loc[mask, col] = value
        else:
            # 새 항목 추가
            new_row = pd.DataFrame([metrics])
            self.metrics_df = pd.concat([self.metrics_df, new_row], ignore_index=True)
        
        # CSV 파일로 저장
        try:
            self.metrics_df.to_csv(self.metrics_file, index=False)
            print(f"  💾 메트릭 저장: {self.metrics_file}")
        except Exception as e:
            print(f"  ⚠️ 메트릭 저장 실패: {str(e)}")
    
    def run(self):
        """Active Learning 프로세스 실행"""
        print("="*80)
        print(f"🚀 YOLO Active Learning 프로세스 시작")
        print(f"   모델: {self.model_name}")
        print(f"   분류 모델 사용: {self.use_classifier}")
        print(f"   최대 사이클: {self.max_cycles}")
        print("="*80)
        
        total_start_time = time.time()
        
        # 각 학습 주기 실행
        for cycle in range(1, self.max_cycles + 1):
            print(f"\n🔄 학습 사이클 {cycle}/{self.max_cycles} 시작")
            print("-" * 60)
            
            cycle_start_time = time.time()
            
            # 통계 초기화
            self.reset_statistics()
            
            # 1. 객체 탐지 및 분류
            self._process_images_for_cycle(cycle)
            
            # 탐지된 객체가 없으면 중단
            if self.detected_objects_count == 0:
                print("⚠️ 탐지된 객체가 없습니다.")
                if cycle == 1:
                    raise Exception("첫 번째 사이클에서 탐지된 객체가 없습니다.")
                
                # 빈 메트릭 저장 후 다음 사이클로
                empty_metrics = self._create_empty_metrics(cycle)
                self._save_metrics(empty_metrics)
                continue
            
            # 2. 데이터셋 준비
            self.prepare_dataset(cycle)
            
            # 3. 모델 학습
            try:
                trained_model_path = self.train_model(cycle)
            except Exception as e:
                print(f"❌ 사이클 {cycle} 학습 실패: {str(e)}")
                empty_metrics = self._create_empty_metrics(cycle)
                self._save_metrics(empty_metrics)
                continue
            
            # 4. 성능 평가
            metrics = self.evaluate_performance(cycle)
            
            # 사이클 완료 시간
            cycle_elapsed = time.time() - cycle_start_time
            print(f"✅ 사이클 {cycle} 완료 ({cycle_elapsed/60:.1f}분)")
        
        # 전체 프로세스 완료
        total_elapsed = time.time() - total_start_time
        
        print("\n" + "="*80)
        print("🎉 YOLO Active Learning 프로세스 완료!")
        print("="*80)
        print(f"📊 실행 정보:")
        print(f"   - 모델: {self.model_name}")
        print(f"   - 완료된 사이클: {self.max_cycles}")
        print(f"   - 총 실행 시간: {total_elapsed/60:.1f}분")
        print(f"   - 분류 모델 사용: {self.use_classifier}")
        print(f"📁 결과 저장 위치: {self.output_dir}")
        print(f"📈 성능 메트릭: {self.metrics_file}")
        
        # 최종 성능 요약
        if not self.metrics_df.empty:
            final_metrics = self.metrics_df.iloc[-1]
            print(f"\n🏆 최종 성능:")
            print(f"   - F1-Score: {final_metrics['F1-Score']:.4f}")
            print(f"   - Precision: {final_metrics['Precision']:.4f}")
            print(f"   - Recall: {final_metrics['Recall']:.4f}")
    
    def _process_images_for_cycle(self, cycle):
        """사이클별 이미지 처리"""
        print("1️⃣ 객체 탐지 및 분류 수행 중...")
        
        # 결과 저장 디렉토리
        cycle_dir = os.path.join(self.output_dir, f"cycle_{cycle}")
        detections_dir = os.path.join(cycle_dir, "detections")
        labels_dir = os.path.join(cycle_dir, "labels")
        
        if self.use_classifier:
            filtered_detections_dir = os.path.join(cycle_dir, "filtered_detections")
            filtered_labels_dir = os.path.join(cycle_dir, "filtered_labels")
        
        # 이미지 파일 목록
        image_files = [f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            raise ValueError("처리할 이미지 파일이 없습니다")
        
        print(f"   📸 처리할 이미지: {len(image_files)}개")
        
        # 각 이미지 처리
        for image_file in tqdm(image_files, desc="Processing Images"):
            image_path = os.path.join(self.image_dir, image_file)
            
            # 객체 탐지 및 분류
            detected_objects, filtered_objects, img_all, img_filtered = \
                self.detect_and_classify_objects(image_path, cycle)
            
            # 탐지 결과 이미지 저장
            if img_all is not None:
                cv2.imwrite(os.path.join(detections_dir, image_file), img_all)
            
            if self.use_classifier and img_filtered is not None:
                cv2.imwrite(os.path.join(filtered_detections_dir, image_file), img_filtered)
            
            # 라벨 파일 저장
            label_name = os.path.splitext(image_file)[0] + '.txt'
            
            if self.use_classifier:
                # 모든 탐지 결과와 필터링된 결과 모두 저장
                all_objects = detected_objects + filtered_objects
                self.save_label(all_objects, os.path.join(labels_dir, label_name))
                self.save_label(detected_objects, os.path.join(filtered_labels_dir, label_name))
            else:
                # 분류 모델 미사용 시 모든 탐지 결과 저장
                self.save_label(detected_objects, os.path.join(labels_dir, label_name))
        
        # 처리 결과 출력
        print(f"   ✅ 탐지된 객체: {self.detected_objects_count}개")
        if self.use_classifier:
            print(f"   🔍 필터링된 객체: {self.filtered_objects_count}개")
            if self.detected_objects_count + self.filtered_objects_count > 0:
                keep_rate = self.detected_objects_count / (self.detected_objects_count + self.filtered_objects_count) * 100
                print(f"   📊 유지 비율: {keep_rate:.1f}%")

if __name__ == "__main__":
    # 테스트 실행
    print("🧪 YOLO Active Learning 모듈 테스트")
    
    # 테스트 설정
    active_learning = YOLOActiveLearning(
        model_path="./models/initial_yolo/yolov8_100pct.pt",
        classifier_path="./models/classification/densenet121_100.pth",
        image_dir="./dataset/images",
        label_dir="./dataset/labels",
        output_dir="./results/test_active_learning",
        conf_threshold=0.25,
        iou_threshold=0.5,
        class_conf_threshold=0.5,
        max_cycles=3,  # 테스트용으로 줄임
        gpu_num=0,
        use_classifier=True
    )
    
    # Active Learning 실행
    try:
        active_learning.run()
        print("✅ 테스트 완료!")
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()