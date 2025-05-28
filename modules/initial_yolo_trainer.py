# modules/initial_yolo_trainer.py
"""
초기 YOLO 모델 학습 모듈
다양한 데이터 비율로 YOLO 모델들을 학습시키는 모듈
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
        초기 YOLO 학습기 초기화
        
        Args:
            dataset_root (str): 데이터셋 루트 디렉토리
            images_dir (str): 원본 이미지 디렉토리
            labels_dir (str): 원본 라벨 디렉토리
            output_dir (str): 모델 저장 디렉토리
            model_type (str): 기본 YOLO 모델 타입
            epochs (int): 학습 에폭 수
            img_size (int): 이미지 크기
            batch_size (int): 배치 크기
            percentages (list): 학습할 데이터 비율 리스트
            train_ratio (float): 훈련 데이터 비율
            valid_ratio (float): 검증 데이터 비율
            test_ratio (float): 테스트 데이터 비율
            random_seed (int): 랜덤 시드
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
        
        # 디렉토리 설정
        self.split_dataset_root = os.path.join(self.dataset_root, 'dataset')
        self.train_dir = os.path.join(self.split_dataset_root, 'train')
        self.valid_dir = os.path.join(self.split_dataset_root, 'valid')
        self.test_dir = os.path.join(self.split_dataset_root, 'test')
        self.temp_dir = os.path.join(self.dataset_root, 'temp_train')
        
        self.setup_directories()
        
    def setup_directories(self):
        """필요한 디렉토리 생성"""
        print("필요한 디렉토리 생성 중...")
        
        # 데이터셋 분할 디렉토리 생성
        for dir_path in [self.train_dir, self.valid_dir, self.test_dir]:
            os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
            os.makedirs(os.path.join(dir_path, 'labels'), exist_ok=True)
        
        # 임시 트레이닝 디렉토리 생성
        os.makedirs(os.path.join(self.temp_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'labels'), exist_ok=True)
        
        # 모델 저장 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("디렉토리 설정 완료")
    
    def split_dataset(self):
        """원본 이미지와 라벨을 train/valid/test로 분할"""
        print("데이터셋 분할 시작...")
        
        # 모든 이미지 파일 가져오기
        image_files = [f for f in os.listdir(self.images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            raise ValueError(f"이미지를 찾을 수 없습니다: {self.images_dir}")
        
        print(f"총 {len(image_files)}개 이미지 파일 발견")
        
        # 일관된 무작위 선택을 위한 시드 설정
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # 이미지 파일 무작위 섞기
        random.shuffle(image_files)
        
        # 데이터셋 분할 계산
        total_images = len(image_files)
        train_size = int(total_images * self.train_ratio)
        valid_size = int(total_images * self.valid_ratio)
        
        # 데이터셋 분할
        train_files = image_files[:train_size]
        valid_files = image_files[train_size:train_size+valid_size]
        test_files = image_files[train_size+valid_size:]
        
        print(f"데이터 분할: Train={len(train_files)}, Valid={len(valid_files)}, Test={len(test_files)}")
        
        # 분할된 디렉토리 정리 (기존 파일 삭제)
        for dir_path in [self.train_dir, self.valid_dir, self.test_dir]:
            for subdir in ['images', 'labels']:
                subdir_path = os.path.join(dir_path, subdir)
                if os.path.exists(subdir_path):
                    for file in os.listdir(subdir_path):
                        os.remove(os.path.join(subdir_path, file))
        
        # 파일 복사 함수
        def copy_files(file_list, source_images, source_labels, dest_dir, split_name):
            copied_count = 0
            for img_file in file_list:
                # 이미지 파일 복사
                src_img = os.path.join(source_images, img_file)
                dst_img = os.path.join(dest_dir, 'images', img_file)
                
                # 레이블 파일명 확인 (확장자를 .txt로 변경)
                label_file = os.path.splitext(img_file)[0] + '.txt'
                src_label = os.path.join(source_labels, label_file)
                dst_label = os.path.join(dest_dir, 'labels', label_file)
                
                # 이미지와 해당 레이블이 모두 존재하는 경우에만 복사
                if os.path.exists(src_img) and os.path.exists(src_label):
                    shutil.copy(src_img, dst_img)
                    
                    # 레이블 파일 복사 (단일 클래스로 변환)
                    with open(src_label, 'r') as original_label:
                        lines = original_label.readlines()
                    
                    with open(dst_label, 'w') as new_label:
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:  # 형식: class_id x y w h
                                # 클래스 ID를 0으로 설정 (단일 클래스)
                                parts[0] = '0'
                                new_label.write(' '.join(parts) + '\n')
                    
                    copied_count += 1
                else:
                    if not os.path.exists(src_img):
                        print(f"경고: 이미지 파일이 없습니다 - {src_img}")
                    if not os.path.exists(src_label):
                        print(f"경고: 라벨 파일이 없습니다 - {src_label}")
            
            print(f"{split_name} 데이터 복사 완료: {copied_count}개")
            return copied_count
        
        # 파일 복사 실행
        train_copied = copy_files(train_files, self.images_dir, self.labels_dir, self.train_dir, "Train")
        valid_copied = copy_files(valid_files, self.images_dir, self.labels_dir, self.valid_dir, "Valid")
        test_copied = copy_files(test_files, self.images_dir, self.labels_dir, self.test_dir, "Test")
        
        print(f"데이터셋 분할 완료:")
        print(f"  - 학습 데이터: {train_copied}개 ({train_copied/total_images*100:.1f}%)")
        print(f"  - 검증 데이터: {valid_copied}개 ({valid_copied/total_images*100:.1f}%)")
        print(f"  - 테스트 데이터: {test_copied}개 ({test_copied/total_images*100:.1f}%)")
        
        if train_copied == 0:
            raise ValueError("훈련 데이터가 없습니다. 이미지와 라벨 파일을 확인하세요.")
        
        return train_copied
    
    def create_subset(self, percentage):
        """주어진 퍼센티지에 맞게 학습 데이터의 일부를 선택 (점층적 방식)"""
        # 학습 디렉토리의 모든 이미지 가져오기
        train_images_dir = os.path.join(self.train_dir, 'images')
        image_files = [f for f in os.listdir(train_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            raise ValueError(f"훈련 이미지를 찾을 수 없습니다: {train_images_dir}")
        
        total_images = len(image_files)
        subset_size = int(total_images * percentage / 100)
        
        print(f"데이터 서브셋 생성: {percentage}% ({subset_size}/{total_images})")
        
        # 일관된 순서를 위한 시드 설정
        random.seed(self.random_seed)
        shuffled_images = random.sample(image_files, len(image_files))
        selected_images = shuffled_images[:subset_size]
        
        # 임시 디렉토리 비우기
        temp_images_dir = os.path.join(self.temp_dir, 'images')
        temp_labels_dir = os.path.join(self.temp_dir, 'labels')
        
        for file in os.listdir(temp_images_dir):
            os.remove(os.path.join(temp_images_dir, file))
        for file in os.listdir(temp_labels_dir):
            os.remove(os.path.join(temp_labels_dir, file))
        
        # 선택된 이미지와 해당 레이블 복사
        copied_count = 0
        for image_file in selected_images:
            # 이미지 복사
            src_img = os.path.join(self.train_dir, 'images', image_file)
            dst_img = os.path.join(temp_images_dir, image_file)
            
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
                
                # 레이블 복사 (같은 파일명에 .txt 확장자)
                label_file = os.path.splitext(image_file)[0] + '.txt'
                src_label = os.path.join(self.train_dir, 'labels', label_file)
                dst_label = os.path.join(temp_labels_dir, label_file)
                
                if os.path.exists(src_label):
                    shutil.copy(src_label, dst_label)
                    copied_count += 1
                else:
                    print(f"경고: 라벨 파일이 없습니다 - {src_label}")
            else:
                print(f"경고: 이미지 파일이 없습니다 - {src_img}")
        
        print(f"서브셋 생성 완료: {copied_count}개 파일 복사됨")
        return copied_count, total_images
    
    def create_dataset_yaml(self):
        """데이터셋 YAML 파일 생성"""
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
        
        print(f"데이터셋 YAML 파일 생성: {yaml_path}")
        return yaml_path
    
    def train_with_percentage(self, percentage):
        """특정 퍼센티지 데이터로 모델 학습"""
        print(f"\n{'='*60}")
        print(f"=== {percentage}% 데이터로 학습 시작 ===")
        print(f"{'='*60}")
        
        # 학습 데이터의 서브셋 생성
        subset_size, total_images = self.create_subset(percentage)
        
        if subset_size == 0:
            print(f"경고: {percentage}% 데이터에 대한 학습 데이터가 없습니다.")
            return None
        
        print(f"전체 {total_images}개 이미지 중 {subset_size}개 선택 ({percentage}%)")
        
        # 데이터셋 YAML 파일 생성
        temp_yaml = self.create_dataset_yaml()
        
        # 모델 초기화
        print(f"YOLO 모델 초기화: {self.model_type}")
        model = YOLO(self.model_type)
        
        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 모델 학습
            print(f"모델 학습 시작...")
            print(f"  - 이미지 수: {subset_size}개 ({percentage}%)")
            print(f"  - 에폭: {self.epochs}")
            print(f"  - 이미지 크기: {self.img_size}")
            print(f"  - 배치 크기: {self.batch_size}")
            
            results = model.train(
                data=temp_yaml,
                epochs=self.epochs,
                imgsz=self.img_size,
                batch=self.batch_size,
                name=f"yolov8_{percentage}pct_{timestamp}",
                patience=15,  # 조기 종료 설정
                save_period=10,  # 10 에폭마다 저장
                plots=True,  # 학습 그래프 저장
                verbose=True
            )
            
            # 학습된 최고 가중치를 출력 디렉토리에 복사
            run_dir = Path(f"runs/detect/yolov8_{percentage}pct_{timestamp}")
            best_weights = run_dir / "weights" / "best.pt"
            
            if best_weights.exists():
                output_path = os.path.join(self.output_dir, f"yolov8_{percentage}pct.pt")
                shutil.copy(best_weights, output_path)
                print(f"✅ 최고 가중치 저장됨: {output_path}")
                
                # 학습 로그도 복사
                results_dir = os.path.join(self.output_dir, f"training_results_{percentage}pct")
                if run_dir.exists():
                    shutil.copytree(run_dir, results_dir, dirs_exist_ok=True)
                    print(f"📊 학습 결과 저장됨: {results_dir}")
                
                return output_path
            else:
                print(f"❌ 경고: {best_weights}에서 최고 가중치를 찾을 수 없습니다")
                return None
                
        except Exception as e:
            print(f"❌ 학습 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_all_percentages(self):
        """모든 퍼센티지에 대해 학습 실행"""
        print("="*80)
        print("초기 YOLO 모델 학습 시작")
        print("="*80)
        
        # 먼저 데이터셋 분할
        print("Step 1: 데이터셋을 train/valid/test로 분할 중...")
        try:
            train_count = self.split_dataset()
        except Exception as e:
            print(f"❌ 데이터셋 분할 실패: {str(e)}")
            return {}
        
        if train_count == 0:
            print("❌ 오류: 학습 데이터가 없습니다. 이미지 및 라벨 디렉토리를 확인하세요.")
            return {}
        
        # 모델 경로를 저장할 딕셔너리
        trained_models = {}
        successful_count = 0
        failed_count = 0
        
        print(f"\nStep 2: 다양한 데이터 비율로 학습 시작")
        print(f"학습할 비율: {self.percentages}")
        
        # 다양한 퍼센티지로 학습 반복
        for i, percentage in enumerate(self.percentages):
            print(f"\n🔄 진행상황: {i+1}/{len(self.percentages)} - {percentage}% 학습")
            
            try:
                model_path = self.train_with_percentage(percentage)
                if model_path and os.path.exists(model_path):
                    trained_models[percentage] = model_path
                    successful_count += 1
                    print(f"✅ {percentage}% 학습 완료: {model_path}")
                else:
                    failed_count += 1
                    print(f"❌ {percentage}% 학습 실패")
            except Exception as e:
                failed_count += 1
                print(f"❌ {percentage}% 학습 중 예외 발생: {str(e)}")
        
        # 임시 디렉토리 정리
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                print("🧹 임시 디렉토리 정리 완료")
        except Exception as e:
            print(f"⚠️ 임시 디렉토리 정리 실패: {str(e)}")
        
        # 결과 요약
        print("\n" + "="*80)
        print("초기 YOLO 모델 학습 완료")
        print("="*80)
        print(f"📊 학습 결과 요약:")
        print(f"  - 성공: {successful_count}개")
        print(f"  - 실패: {failed_count}개") 
        print(f"  - 총 시도: {len(self.percentages)}개")
        print(f"  - 모델 저장 위치: {self.output_dir}")
        
        if trained_models:
            print(f"\n✅ 성공적으로 학습된 모델:")
            for percentage, path in trained_models.items():
                print(f"  - {percentage}%: {path}")
        
        return trained_models

if __name__ == "__main__":
    # 테스트 실행
    trainer = InitialYOLOTrainer(
        dataset_root='./dataset',
        images_dir='./dataset/images',
        labels_dir='./dataset/labels',
        output_dir='./models/initial_yolo',
        model_type='yolov8n.pt',
        epochs=50,  # 테스트용으로 낮춘 값
        percentages=[10, 50, 100]  # 테스트용으로 줄인 값
    )
    
    results = trainer.train_all_percentages()
    print(f"학습 완료! 결과: {results}")