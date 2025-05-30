# modules/manual_labeling_ui.py
"""
수동 라벨링 UI 모듈
YOLO로 탐지된 객체들을 사용자가 수동으로 Class 0/1로 분류하는 GUI 제공
"""

import os
import cv2
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
from tqdm import tqdm

class ManualLabelingUI:
    """
    수동 라벨링을 위한 GUI 클래스
    YOLO로 탐지된 객체들을 사용자가 직접 분류
    """
    
    def __init__(self, yolo_model_path, images_dir, output_dir, conf_threshold=0.25, iou_threshold=0.5):
        """
        수동 라벨링 UI 초기화
        
        Args:
            yolo_model_path (str): YOLO 모델 경로
            images_dir (str): 이미지 디렉토리 경로
            output_dir (str): 결과 저장 디렉토리 경로
            conf_threshold (float): 객체 검출 신뢰도 임계값
            iou_threshold (float): IoU 임계값
        """
        self.yolo_model_path = yolo_model_path
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # YOLO 모델 로드
        print(f"📥 YOLO 모델 로딩: {yolo_model_path}")
        self.model = YOLO(yolo_model_path)
        
        # 출력 디렉토리 설정
        self.class0_dir = os.path.join(output_dir, 'class0')
        self.class1_dir = os.path.join(output_dir, 'class1')
        os.makedirs(self.class0_dir, exist_ok=True)
        os.makedirs(self.class1_dir, exist_ok=True)
        
        # 이미지 파일 목록
        self.image_files = [f for f in os.listdir(images_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not self.image_files:
            raise ValueError(f"이미지 파일을 찾을 수 없습니다: {images_dir}")
        
        print(f"📊 처리할 이미지: {len(self.image_files)}개")
        
        # 상태 변수
        self.current_image_idx = 0
        self.current_objects = []
        self.current_object_idx = 0
        self.total_labeled = 0
        
        # UI 컴포넌트
        self.root = None
        self.canvas = None
        self.info_label = None
        self.progress_label = None
        self.photo = None
        
        # 첫 번째 이미지의 객체들 미리 추출
        self._load_current_image_objects()
        
    def _load_current_image_objects(self):
        """현재 이미지의 객체들 로드"""
        if self.current_image_idx >= len(self.image_files):
            self.current_objects = []
            return
            
        image_filename = self.image_files[self.current_image_idx]
        image_path = os.path.join(self.images_dir, image_filename)
        
        # YOLO 추론 실행
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"⚠️ 이미지 로드 실패: {image_path}")
                self.current_objects = []
                return
            
            results = self.model.predict(
                source=img,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                save=False,
                verbose=False
            )
            
            # 탐지된 객체들 추출
            self.current_objects = []
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
                            self.current_objects.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': conf,
                                'image': obj_img,
                                'index': i,
                                'labeled': False
                            })
            
            print(f"  📦 {image_filename}: {len(self.current_objects)}개 객체 탐지됨")
            
        except Exception as e:
            print(f"⚠️ 객체 탐지 실패 ({image_filename}): {str(e)}")
            self.current_objects = []
    
    def setup_ui(self):
        """UI 설정"""
        self.root = tk.Tk()
        self.root.title("Manual Object Labeling - YOLO Active Learning")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 상단 정보 프레임
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 진행 정보 라벨
        self.progress_label = ttk.Label(info_frame, text="", font=("Arial", 12, "bold"))
        self.progress_label.pack()
        
        # 상세 정보 라벨
        self.info_label = ttk.Label(info_frame, text="", font=("Arial", 10))
        self.info_label.pack()
        
        # 이미지 표시 프레임
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 캔버스 (스크롤 지원)
        canvas_frame = ttk.Frame(image_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='white', highlightthickness=1, highlightbackground='gray')
        scrollbar_v = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        scrollbar_h = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar_v.pack(side="right", fill="y")
        scrollbar_h.pack(side="bottom", fill="x")
        
        # 하단 버튼 프레임
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # 분류 버튼들 (크고 눈에 띄게)
        classify_frame = ttk.Frame(button_frame)
        classify_frame.pack(pady=(0, 10))
        
        # Class 0 버튼 (초록색)
        self.class0_btn = tk.Button(classify_frame, text="✅ Class 0 (Keep)\n유지할 객체", 
                                   command=lambda: self.label_object(0),
                                   font=("Arial", 14, "bold"), 
                                   bg='#4CAF50', fg='white', 
                                   width=20, height=3,
                                   relief='raised', bd=3)
        self.class0_btn.pack(side=tk.LEFT, padx=10)
        
        # Class 1 버튼 (빨간색)
        self.class1_btn = tk.Button(classify_frame, text="❌ Class 1 (Filter)\n필터링할 객체", 
                                   command=lambda: self.label_object(1),
                                   font=("Arial", 14, "bold"), 
                                   bg='#f44336', fg='white', 
                                   width=20, height=3,
                                   relief='raised', bd=3)
        self.class1_btn.pack(side=tk.LEFT, padx=10)
        
        # 네비게이션 버튼들
        nav_frame = ttk.Frame(button_frame)
        nav_frame.pack()
        
        ttk.Button(nav_frame, text="⬅️ 이전 객체", 
                  command=self.prev_object).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="➡️ 다음 객체", 
                  command=self.next_object).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="⬆️ 이전 이미지", 
                  command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="⬇️ 다음 이미지", 
                  command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        # 유틸리티 버튼들
        util_frame = ttk.Frame(button_frame)
        util_frame.pack(pady=(10, 0))
        
        ttk.Button(util_frame, text="🔄 새로고침", 
                  command=self.refresh_display).pack(side=tk.LEFT, padx=5)
        ttk.Button(util_frame, text="📊 통계 보기", 
                  command=self.show_statistics).pack(side=tk.LEFT, padx=5)
        ttk.Button(util_frame, text="✅ 라벨링 완료", 
                  command=self.finish_labeling).pack(side=tk.RIGHT, padx=5)
        
        # 키보드 단축키 바인딩
        self.root.bind('<Key-1>', lambda e: self.label_object(0))
        self.root.bind('<Key-2>', lambda e: self.label_object(1))
        self.root.bind('<Left>', lambda e: self.prev_object())
        self.root.bind('<Right>', lambda e: self.next_object())
        self.root.bind('<Up>', lambda e: self.prev_image())
        self.root.bind('<Down>', lambda e: self.next_image())
        self.root.bind('<Escape>', lambda e: self.finish_labeling())
        
        # 포커스 설정 (키보드 이벤트를 받기 위해)
        self.root.focus_set()
        
        print("✅ UI 설정 완료")
        print("📝 단축키:")
        print("  - 1: Class 0 (Keep)")
        print("  - 2: Class 1 (Filter)")
        print("  - 방향키: 네비게이션")
        print("  - ESC: 완료")
    
    def update_display(self):
        """현재 이미지와 객체 표시 업데이트"""
        if self.current_image_idx >= len(self.image_files):
            self.finish_labeling()
            return
        
        # 현재 이미지 정보
        image_filename = self.image_files[self.current_image_idx]
        image_path = os.path.join(self.images_dir, image_filename)
        
        # 진행상황 업데이트
        total_images = len(self.image_files)
        total_objects = len(self.current_objects)
        
        if total_objects == 0:
            # 객체가 없는 경우 다음 이미지로
            self.next_image()
            return
        
        progress_text = (f"이미지 {self.current_image_idx + 1}/{total_images} | "
                        f"객체 {self.current_object_idx + 1}/{total_objects} | "
                        f"라벨링됨: {self.total_labeled}개")
        self.progress_label.config(text=progress_text)
        
        # 현재 객체 정보
        current_obj = self.current_objects[self.current_object_idx]
        info_text = (f"파일: {image_filename} | "
                    f"신뢰도: {current_obj['confidence']:.3f} | "
                    f"상태: {'✅ 완료' if current_obj['labeled'] else '⏳ 대기'}")
        self.info_label.config(text=info_text)
        
        # 이미지 로드 및 표시
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"⚠️ 이미지 로드 실패: {image_path}")
                return
            
            # 이미지에 바운딩 박스들 그리기
            img_display = img.copy()
            
            for i, obj in enumerate(self.current_objects):
                bbox = obj['bbox']
                x1, y1, x2, y2 = bbox
                
                # 현재 객체는 초록색, 나머지는 회색
                if i == self.current_object_idx:
                    color = (0, 255, 0)  # 초록색
                    thickness = 3
                elif obj['labeled']:
                    color = (128, 128, 128)  # 회색 (완료됨)
                    thickness = 1
                else:
                    color = (200, 200, 200)  # 연한 회색
                    thickness = 1
                
                cv2.rectangle(img_display, (x1, y1), (x2, y2), color, thickness)
                
                # 객체 번호 표시
                label_text = f"#{i+1}"
                if obj['labeled']:
                    label_text += " ✓"
                
                cv2.putText(img_display, label_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 현재 객체 확대 표시 (오른쪽 상단)
            current_obj = self.current_objects[self.current_object_idx]
            obj_img = current_obj['image']
            
            # 객체 이미지 크기 조정 (최대 200x200)
            h, w = obj_img.shape[:2]
            max_size = 200
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                obj_img_resized = cv2.resize(obj_img, (new_w, new_h))
            else:
                obj_img_resized = obj_img
            
            # 객체 이미지를 메인 이미지 오른쪽 상단에 합성
            oh, ow = obj_img_resized.shape[:2]
            img_h, img_w = img_display.shape[:2]
            
            # 여백 확보
            margin = 10
            if img_w > ow + margin and img_h > oh + margin:
                # 배경 박스 그리기
                cv2.rectangle(img_display, 
                             (img_w - ow - margin, margin), 
                             (img_w - margin, oh + margin + 30), 
                             (255, 255, 255), -1)
                cv2.rectangle(img_display, 
                             (img_w - ow - margin, margin), 
                             (img_w - margin, oh + margin + 30), 
                             (0, 0, 0), 2)
                
                # 객체 이미지 합성
                img_display[margin:margin+oh, img_w-ow-margin:img_w-margin] = obj_img_resized
                
                # "현재 객체" 텍스트
                cv2.putText(img_display, "Current Object", 
                           (img_w - ow - margin, margin + oh + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # OpenCV → PIL → PhotoImage 변환
            img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # 캔버스 크기에 맞게 조정
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # 이미지 크기 조정
                img_w, img_h = img_pil.size
                scale = min(canvas_width/img_w, canvas_height/img_h) * 0.9
                
                if scale < 1:
                    new_w, new_h = int(img_w * scale), int(img_h * scale)
                    img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # PhotoImage 생성 및 표시
            self.photo = ImageTk.PhotoImage(img_pil)
            
            # 캔버스 클리어 및 이미지 표시
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, 
                                   image=self.photo, anchor=tk.CENTER)
            
            # 스크롤 영역 업데이트
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            print(f"⚠️ 이미지 표시 실패: {str(e)}")
    
    def save_object(self, obj_img, class_label, image_filename, obj_idx):
        """객체 이미지를 해당 클래스 디렉토리에 저장"""
        output_dir = self.class0_dir if class_label == 0 else self.class1_dir
        
        # 파일명 생성: 원본이미지명_객체인덱스.jpg
        base_name = os.path.splitext(image_filename)[0]
        obj_filename = f"{base_name}_obj_{obj_idx:03d}.jpg"
        obj_path = os.path.join(output_dir, obj_filename)
        
        # 이미지 저장
        try:
            cv2.imwrite(obj_path, obj_img)
            print(f"  💾 저장: {obj_filename} → Class {class_label}")
            return True
        except Exception as e:
            print(f"  ❌ 저장 실패: {obj_filename} - {str(e)}")
            return False
    
    def label_object(self, class_label):
        """현재 객체에 라벨 지정"""
        if not self.current_objects or self.current_object_idx >= len(self.current_objects):
            return
        
        current_obj = self.current_objects[self.current_object_idx]
        
        # 이미 라벨링된 객체는 스킵
        if current_obj['labeled']:
            self.next_object()
            return
        
        image_filename = self.image_files[self.current_image_idx]
        
        # 객체 저장
        success = self.save_object(
            current_obj['image'], 
            class_label, 
            image_filename, 
            current_obj['index']
        )
        
        if success:
            # 라벨링 완료 표시
            current_obj['labeled'] = True
            self.total_labeled += 1
            
            # 다음 객체로 자동 이동
            self.next_object()
    
    def next_object(self):
        """다음 객체로 이동"""
        if not self.current_objects:
            self.next_image()
            return
        
        # 다음 라벨링되지 않은 객체 찾기
        start_idx = self.current_object_idx
        while True:
            self.current_object_idx = (self.current_object_idx + 1) % len(self.current_objects)
            
            # 한 바퀴 돌았으면 다음 이미지로
            if self.current_object_idx == start_idx:
                self.next_image()
                break
            
            # 라벨링되지 않은 객체 발견
            if not self.current_objects[self.current_object_idx]['labeled']:
                break
        
        self.update_display()
    
    def prev_object(self):
        """이전 객체로 이동"""
        if not self.current_objects:
            return
        
        self.current_object_idx = (self.current_object_idx - 1) % len(self.current_objects)
        self.update_display()
    
    def next_image(self):
        """다음 이미지로 이동"""
        self.current_image_idx += 1
        self.current_object_idx = 0
        
        if self.current_image_idx >= len(self.image_files):
            self.finish_labeling()
            return
        
        # 새 이미지의 객체들 로드
        self._load_current_image_objects()
        self.update_display()
    
    def prev_image(self):
        """이전 이미지로 이동"""
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.current_object_idx = 0
            self._load_current_image_objects()
            self.update_display()
    
    def refresh_display(self):
        """화면 새로고침"""
        self.update_display()
    
    def show_statistics(self):
        """라벨링 통계 표시"""
        class0_count = len(os.listdir(self.class0_dir))
        class1_count = len(os.listdir(self.class1_dir))
        total_labeled = class0_count + class1_count
        
        # 현재 이미지까지의 총 객체 수 추정
        processed_images = self.current_image_idx
        avg_objects_per_image = self.total_labeled / max(1, processed_images) if processed_images > 0 else 0
        
        stats_message = f"""📊 라벨링 통계
        
✅ Class 0 (Keep): {class0_count}개
❌ Class 1 (Filter): {class1_count}개
📊 총 라벨링: {total_labeled}개

📸 처리된 이미지: {processed_images}/{len(self.image_files)}
📦 평균 객체/이미지: {avg_objects_per_image:.1f}개

진행률: {processed_images/len(self.image_files)*100:.1f}%"""
        
        messagebox.showinfo("라벨링 통계", stats_message)
    
    def finish_labeling(self):
        """라벨링 완료"""
        class0_count = len(os.listdir(self.class0_dir))
        class1_count = len(os.listdir(self.class1_dir))
        total_count = class0_count + class1_count
        
        if total_count == 0:
            result = messagebox.askyesno("경고", 
                                       "라벨링된 객체가 없습니다.\n"
                                       "정말 종료하시겠습니까?")
            if not result:
                return
        
        completion_message = f"""🎉 라벨링 완료!

📊 최종 결과:
  ✅ Class 0 (Keep): {class0_count}개
  ❌ Class 1 (Filter): {class1_count}개
  📊 총합: {total_count}개

💾 저장 위치:
  📁 {self.output_dir}

이 데이터는 Classification 모델 학습에 사용됩니다."""
        
        messagebox.showinfo("완료", completion_message)
        
        if self.root:
            self.root.quit()
            self.root.destroy()
    
    def run(self):
        """UI 실행"""
        if not self.image_files:
            print("❌ 라벨링할 이미지가 없습니다.")
            return None
        
        print(f"🏷️ 수동 라벨링 UI 시작")
        print(f"  - 총 이미지: {len(self.image_files)}개")
        print(f"  - 첫 이미지 객체: {len(self.current_objects)}개")
        
        # UI 설정 및 실행
        self.setup_ui()
        
        # 초기 화면 표시 (UI가 렌더링된 후)
        self.root.after(500, self.update_display)
        
        try:
            # 메인 루프 실행
            self.root.mainloop()
        except Exception as e:
            print(f"⚠️ UI 실행 중 오류: {str(e)}")
        
        # 결과 검증
        class0_count = len(os.listdir(self.class0_dir))
        class1_count = len(os.listdir(self.class1_dir))
        
        if class0_count > 0 or class1_count > 0:
            print(f"✅ 라벨링 완료: Class 0={class0_count}개, Class 1={class1_count}개")
            return self.output_dir
        else:
            print("⚠️ 라벨링된 데이터가 없습니다.")
            return None

if __name__ == "__main__":
    # 테스트 실행
    print("🧪 Manual Labeling UI 테스트")
    
    # 테스트 설정
    ui = ManualLabelingUI(
        yolo_model_path="./results/01_initial_yolo/yolov8_100pct.pt",
        images_dir="./dataset/images",
        output_dir="./test_manual_labeling",
        conf_threshold=0.25,
        iou_threshold=0.5
    )
    
    # UI 실행
    result = ui.run()
    
    if result:
        print(f"✅ 테스트 완료: {result}")
    else:
        print("❌ 테스트 실패")