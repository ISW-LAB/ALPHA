# main.py (Fixed)
"""
데이터 비율 설정에 유연하게 대응하는 수정된 main.py
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

# 모듈 임포트
from modules.initial_yolo_trainer import InitialYOLOTrainer
from modules.classification_trainer import ClassificationTrainer
from modules.iterative_processor import IterativeProcessor
from modules.object_classifier import ObjectClassifier
from modules.yolo_active_learning import YOLOActiveLearning

class CompletePipeline:
    """완전한 파이프라인 관리 클래스 (유연한 데이터 비율 지원)"""
    
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        
        print("🔧 파이프라인 설정:")
        print(f"  - 이미지 디렉토리: {config['images_dir']}")
        print(f"  - 라벨 디렉토리: {config['labels_dir']}")
        print(f"  - 결과 저장: {config['iterative_output']}")
        print(f"  - 데이터 비율: {config['data_percentages']}")
        print(f"  - GPU: {config['gpu_num']}")
        
    def setup_directories(self):
        """필요한 디렉토리 생성"""
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
        """스텝 1: 초기 YOLO 모델 학습"""
        print("="*80)
        print("STEP 1: 초기 YOLO 모델 학습")
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
        
        # 가장 높은 비율의 모델 선택 (100%가 없으면 최대값)
        max_percentage = max(self.config['data_percentages'])
        best_model_path = trained_models.get(max_percentage)
        
        if not best_model_path or not os.path.exists(best_model_path):
            # 실제로 존재하는 모델 중에서 선택
            available_models = {k: v for k, v in trained_models.items() if v and os.path.exists(v)}
            
            if not available_models:
                raise Exception("학습된 YOLO 모델이 없습니다.")
            
            # 가장 높은 비율의 모델 선택
            max_available_percentage = max(available_models.keys())
            best_model_path = available_models[max_available_percentage]
            
            print(f"📝 최대 비율 {max_percentage}% 모델이 없어서 {max_available_percentage}% 모델을 사용합니다.")
        
        print(f"✅ 초기 YOLO 모델 학습 완료: {best_model_path}")
        return best_model_path
    
    def step2_first_inference_and_manual_labeling(self, yolo_model_path):
        """스텝 2: 초기 모델로 첫 추론 + 수동 라벨링 (GUI 대안 포함)"""
        print("="*80)
        print("STEP 2: 초기 모델 첫 추론 + 수동 라벨링")
        print("="*80)
        
        # 기존 라벨링 데이터 확인
        class0_dir = os.path.join(self.config['manual_labeling_output'], 'class0')
        class1_dir = os.path.join(self.config['manual_labeling_output'], 'class1')
        
        if (os.path.exists(class0_dir) and os.path.exists(class1_dir) and 
            len(os.listdir(class0_dir)) > 0 and len(os.listdir(class1_dir)) > 0):
            print("📂 기존 수동 라벨링 데이터 발견!")
            print(f"  - Class 0: {len(os.listdir(class0_dir))}개")
            print(f"  - Class 1: {len(os.listdir(class1_dir))}개")
            
            use_existing = input("기존 라벨링 데이터를 사용하시겠습니까? (y/n): ").lower().strip()
            if use_existing == 'y':
                return self.config['manual_labeling_output']
        
        # 첫 추론 실행
        print("🔍 초기 모델로 첫 추론 실행...")
        self._perform_first_inference(yolo_model_path)
        
        # 라벨링 방법 선택
        print("\n🏷️ 수동 라벨링 방법 선택:")
        print("1. GUI 라벨링 (권장)")
        print("2. CLI 라벨링 (터미널 기반)")
        print("3. 배치 라벨링 (파일 기반)")
        print("4. 자동 라벨링 (신뢰도 기반)")
        
        while True:
            choice = input("선택 (1-4): ").strip()
            
            if choice == '1':
                # GUI 라벨링 시도
                try:
                    return self._try_gui_labeling(yolo_model_path)
                except Exception as e:
                    print(f"❌ GUI 라벨링 실패: {str(e)}")
                    print("다른 방법을 선택하세요.")
                    continue
                    
            elif choice == '2':
                # CLI 라벨링
                return self._try_cli_labeling(yolo_model_path)
                
            elif choice == '3':
                # 배치 라벨링
                return self._try_batch_labeling(yolo_model_path)
                
            elif choice == '4':
                # 자동 라벨링
                return self._try_auto_labeling(yolo_model_path)
                
            else:
                print("잘못된 선택입니다. 1-4 중에서 선택하세요.")
    
    def _perform_first_inference(self, yolo_model_path):
        """초기 모델로 첫 추론 실행"""
        from ultralytics import YOLO
        import cv2
        from tqdm import tqdm
        
        model = YOLO(yolo_model_path)
        inference_dir = self.config['first_inference_output']
        os.makedirs(inference_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(self.config['images_dir']) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"  📸 {len(image_files)}개 이미지에 대해 첫 추론 실행...")
        
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
        
        print(f"  ✅ 첫 추론 완료: 총 {total_detections}개 객체 탐지됨")
    
    def _try_gui_labeling(self, yolo_model_path):
        """GUI 라벨링 시도"""
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
                raise Exception("GUI 라벨링이 완료되지 않았습니다.")
            
            print(f"✅ GUI 라벨링 완료: {labeled_data_path}")
            return labeled_data_path
            
        except ImportError as e:
            if "tkinter" in str(e).lower():
                print("❌ GUI 라이브러리(tkinter)가 설치되지 않았습니다.")
                print("Ubuntu/Debian: sudo apt-get install python3-tk")
                print("CentOS/RHEL: sudo yum install tkinter")
            else:
                print(f"❌ GUI 모듈 임포트 실패: {str(e)}")
            raise
        except Exception as e:
            print(f"❌ GUI 실행 실패: {str(e)}")
            raise
    
    def _try_auto_labeling(self, yolo_model_path):
        """자동 라벨링 시도 (간단한 구현)"""
        print("🤖 자동 라벨링 실행 중...")
        
        # 자동 라벨링 임계값 설정
        threshold_input = input("자동 분류 임계값을 입력하세요 (0.3-0.9, 기본값: 0.6): ").strip()
        try:
            threshold = float(threshold_input) if threshold_input else 0.6
        except ValueError:
            threshold = 0.6
        
        print(f"  📊 임계값: {threshold} (≥ {threshold}: Keep, < {threshold}: Filter)")
        
        from ultralytics import YOLO
        import cv2
        from tqdm import tqdm
        
        # YOLO 모델 로드
        model = YOLO(yolo_model_path)
        
        # 출력 디렉토리
        class0_dir = os.path.join(self.config['manual_labeling_output'], 'class0')
        class1_dir = os.path.join(self.config['manual_labeling_output'], 'class1')
        os.makedirs(class0_dir, exist_ok=True)
        os.makedirs(class1_dir, exist_ok=True)
        
        # 이미지 파일 목록
        image_files = [f for f in os.listdir(self.config['images_dir']) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        class0_count = 0
        class1_count = 0
        
        for image_file in tqdm(image_files, desc="자동 라벨링"):
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
                        
                        # 좌표 정수 변환 및 경계 확인
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        h, w = img.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        # 유효한 바운딩 박스인지 확인
                        if x2 > x1 and y2 > y1:
                            # 객체 이미지 추출
                            obj_img = img[y1:y2, x1:x2]
                            
                            if obj_img.size > 0:
                                # 신뢰도에 따른 자동 분류
                                if conf >= threshold:
                                    # Class 0 (Keep)
                                    output_dir = class0_dir
                                    class0_count += 1
                                else:
                                    # Class 1 (Filter)
                                    output_dir = class1_dir
                                    class1_count += 1
                                
                                # 파일명 생성 및 저장
                                base_name = os.path.splitext(image_file)[0]
                                obj_filename = f"{base_name}_obj_{i:03d}_conf_{conf:.3f}.jpg"
                                obj_path = os.path.join(output_dir, obj_filename)
                                
                                cv2.imwrite(obj_path, obj_img)
                                
            except Exception as e:
                print(f"⚠️ 오류 ({image_file}): {str(e)}")
        
        total_objects = class0_count + class1_count
        
        print(f"\n🎉 자동 라벨링 완료!")
        print(f"📊 결과:")
        print(f"   Class 0 (Keep): {class0_count}개")
        print(f"   Class 1 (Filter): {class1_count}개")
        print(f"   총 객체: {total_objects}개")
        print(f"💾 저장 위치: {self.config['manual_labeling_output']}")
        
        if total_objects > 0:
            return self.config['manual_labeling_output']
        else:
            raise Exception("자동 라벨링된 객체가 없습니다.")
    
    def _try_cli_labeling(self, yolo_model_path):
        """CLI 라벨링 (간단한 구현)"""
        print("💻 CLI 라벨링은 현재 자동 라벨링으로 대체됩니다.")
        return self._try_auto_labeling(yolo_model_path)
    
    def _try_batch_labeling(self, yolo_model_path):
        """배치 라벨링 (간단한 구현)"""
        print("📁 배치 라벨링은 현재 자동 라벨링으로 대체됩니다.")
        return self._try_auto_labeling(yolo_model_path)
    
    def step3_classification_training(self, labeled_data_path):
        """스텝 3: Classification 모델 학습"""
        print("="*80)
        print("STEP 3: Classification 모델 학습")
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
        
        # 가장 높은 비율의 모델 선택
        max_ratio = max(self.config['classification_ratios'])
        best_classifier_path = os.path.join(
            self.config['classification_output'], 
            f'densenet121_{int(max_ratio*100)}.pth'
        )
        
        if not os.path.exists(best_classifier_path):
            # 실제로 존재하는 분류 모델 찾기
            import glob
            available_models = glob.glob(os.path.join(self.config['classification_output'], 'densenet121_*.pth'))
            
            if not available_models:
                raise Exception("Classification 모델 학습에 실패했습니다.")
            
            # 가장 최근 모델 선택
            best_classifier_path = max(available_models, key=os.path.getctime)
            print(f"📝 최대 비율 {max_ratio*100}% 모델이 없어서 {os.path.basename(best_classifier_path)}를 사용합니다.")
        
        print(f"✅ Classification 모델 학습 완료: {best_classifier_path}")
        return best_classifier_path
    
    def step4_iterative_process(self, classifier_path):
        """스텝 4: Iterative Process 실행"""
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
        
        print("✅ Iterative Process 완료!")
        return results
    
    def run_complete_pipeline(self):
        """완전한 파이프라인 실행"""
        total_start_time = time.time()
        
        try:
            print("🚀 Complete Pipeline 시작!")
            print("="*80)
            
            # Step 1: 초기 YOLO 학습
            yolo_model_path = self.step1_initial_yolo_training()
            
            # Step 2: 첫 추론 + 수동 라벨링
            labeled_data_path = self.step2_first_inference_and_manual_labeling(yolo_model_path)
            
            # Step 3: Classification 학습
            classifier_path = self.step3_classification_training(labeled_data_path)
            
            # Step 4: Iterative Process
            final_results = self.step4_iterative_process(classifier_path)
            
            # 총 실행 시간
            total_elapsed = time.time() - total_start_time
            
            print("="*80)
            print("🎉 전체 파이프라인 완료!")
            print("="*80)
            print(f"⏰ 총 실행 시간: {total_elapsed/60:.1f}분")
            print(f"📊 결과 요약:")
            print(f"  - 초기 YOLO 모델: {yolo_model_path}")
            print(f"  - 수동 라벨링 데이터: {labeled_data_path}")
            print(f"  - Classification 모델: {classifier_path}")
            print(f"  - 최종 결과: {self.config['iterative_output']}")
            
            if final_results and "summary" in final_results:
                summary = final_results["summary"]
                print(f"📈 Iterative Process 결과:")
                print(f"  - 성공률: {summary['success_rate']:.1f}%")
                print(f"  - 처리된 모델: {summary['total_models']}개")
            
            return final_results
            
        except Exception as e:
            print(f"❌ 파이프라인 실행 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_single_step(self, step_num):
        """특정 단계만 실행"""
        if step_num == 1:
            return self.step1_initial_yolo_training()
        elif step_num == 2:
            # 사용 가능한 YOLO 모델 찾기
            import glob
            model_files = glob.glob(os.path.join(self.config['initial_yolo_output'], 'yolov8_*.pt'))
            
            if not model_files:
                print(f"❌ YOLO 모델을 찾을 수 없습니다: {self.config['initial_yolo_output']}")
                print("먼저 Step 1을 실행하세요.")
                return None
            
            # 가장 최근 모델 또는 가장 높은 비율 모델 선택
            yolo_model = max(model_files, key=os.path.getctime)
            print(f"📝 사용할 YOLO 모델: {os.path.basename(yolo_model)}")
            
            return self.step2_first_inference_and_manual_labeling(yolo_model)
        elif step_num == 3:
            labeled_data = self.config['manual_labeling_output']
            if not os.path.exists(os.path.join(labeled_data, 'class0')):
                print(f"❌ 수동 라벨링 데이터를 찾을 수 없습니다: {labeled_data}")
                print("먼저 Step 2를 실행하세요.")
                return None
            return self.step3_classification_training(labeled_data)
        elif step_num == 4:
            # 사용 가능한 분류 모델 찾기
            import glob
            classifier_files = glob.glob(os.path.join(self.config['classification_output'], 'densenet121_*.pth'))
            
            if not classifier_files:
                print(f"❌ Classification 모델을 찾을 수 없습니다: {self.config['classification_output']}")
                print("먼저 Step 3을 실행하세요.")
                return None
            
            # 가장 최근 모델 선택
            classifier = max(classifier_files, key=os.path.getctime)
            print(f"📝 사용할 Classification 모델: {os.path.basename(classifier)}")
            
            return self.step4_iterative_process(classifier)
        else:
            print(f"❌ 유효하지 않은 단계 번호: {step_num}")
            return None

def create_default_config():
    """기본 설정 생성"""
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
    """설정 파일 로드"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ 설정 파일 로드: {config_path}")
        return config
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")
        return None

def save_config(config, config_path):
    """설정 파일 저장"""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✅ 설정 파일 저장: {config_path}")
    except Exception as e:
        print(f"❌ 설정 파일 저장 실패: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Complete Pipeline for YOLO + Classification Active Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py                              # 전체 파이프라인 실행
  python main.py --step 1                     # YOLO 학습만
  python main.py --step 2                     # 첫 추론 + 수동 라벨링만
  python main.py --step 3                     # Classification 학습만
  python main.py --step 4                     # Iterative Process만
  python main.py --config my_config.json      # 설정 파일 사용
  python main.py --create-config config.json  # 기본 설정 파일 생성

라벨링 방법:
  1. GUI 라벨링: 직관적인 그래픽 인터페이스 (권장)
  2. CLI 라벨링: 터미널에서 대화형 라벨링
  3. 배치 라벨링: 파일 탐색기로 수동 분류
  4. 자동 라벨링: 신뢰도 기반 자동 분류
        """
    )
    
    parser.add_argument('--config', type=str, help='설정 파일 경로')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4], 
                       help='실행할 단계 (1: YOLO학습, 2: 수동라벨링, 3: Classification, 4: Iterative)')
    parser.add_argument('--images_dir', type=str, help='이미지 디렉토리 경로')
    parser.add_argument('--labels_dir', type=str, help='라벨 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, help='출력 디렉토리 경로')
    parser.add_argument('--gpu_num', type=int, help='사용할 GPU 번호')
    parser.add_argument('--create-config', type=str, help='기본 설정 파일을 지정된 경로에 생성')
    
    args = parser.parse_args()
    
    # 기본 설정 파일 생성
    if args.create_config:
        config = create_default_config()
        save_config(config, args.create_config)
        print(f"✅ 기본 설정 파일이 생성되었습니다: {args.create_config}")
        print("설정을 수정한 후 다시 실행하세요.")
        return
    
    # 설정 로드
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        if config is None:
            return
    else:
        config = create_default_config()
        if args.config:
            print(f"⚠️ 설정 파일을 찾을 수 없습니다: {args.config}")
            print("기본 설정을 사용합니다.")
    
    # 명령행 인수로 설정 오버라이드
    if args.images_dir:
        config['images_dir'] = args.images_dir
    if args.labels_dir:
        config['labels_dir'] = args.labels_dir
    if args.output_dir:
        config['iterative_output'] = args.output_dir
    if args.gpu_num is not None:
        config['gpu_num'] = args.gpu_num
    
    # 입력 검증
    if not os.path.exists(config['images_dir']):
        print(f"❌ 이미지 디렉토리를 찾을 수 없습니다: {config['images_dir']}")
        return
    
    if not os.path.exists(config['labels_dir']):
        print(f"❌ 라벨 디렉토리를 찾을 수 없습니다: {config['labels_dir']}")
        return
    
    # 파이프라인 실행
    pipeline = CompletePipeline(config)
    
    if args.step:
        # 특정 단계만 실행
        print(f"🎯 Step {args.step} 실행")
        result = pipeline.run_single_step(args.step)
        if result:
            print(f"✅ Step {args.step} 완료")
        else:
            print(f"❌ Step {args.step} 실패")
    else:
        # 전체 파이프라인 실행
        print("🚀 전체 파이프라인 실행")
        result = pipeline.run_complete_pipeline()
        if result:
            print("✅ 전체 파이프라인 완료")
        else:
            print("❌ 파이프라인 실행 실패")

if __name__ == "__main__":
    main()