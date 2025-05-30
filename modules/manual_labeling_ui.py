# modules/manual_labeling_ui.py
"""
Manual Labeling UI Module
Provides GUI for users to manually classify objects detected by YOLO into Class 0/1
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
    GUI class for manual labeling
    Users directly classify objects detected by YOLO
    """
    
    def __init__(self, yolo_model_path, images_dir, output_dir, conf_threshold=0.25, iou_threshold=0.5):
        """
        Initialize Manual Labeling UI
        
        Args:
            yolo_model_path (str): YOLO model path
            images_dir (str): Image directory path
            output_dir (str): Results save directory path
            conf_threshold (float): Object detection confidence threshold
            iou_threshold (float): IoU threshold
        """
        self.yolo_model_path = yolo_model_path
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load YOLO model
        print(f"📥 Loading YOLO model: {yolo_model_path}")
        self.model = YOLO(yolo_model_path)
        
        # Set output directories
        self.class0_dir = os.path.join(output_dir, 'class0')
        self.class1_dir = os.path.join(output_dir, 'class1')
        os.makedirs(self.class0_dir, exist_ok=True)
        os.makedirs(self.class1_dir, exist_ok=True)
        
        # Image file list
        self.image_files = [f for f in os.listdir(images_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not self.image_files:
            raise ValueError(f"No image files found: {images_dir}")
        
        print(f"📊 Images to process: {len(self.image_files)}")
        
        # State variables
        self.current_image_idx = 0
        self.current_objects = []
        self.current_object_idx = 0
        self.total_labeled = 0
        
        # UI components
        self.root = None
        self.canvas = None
        self.info_label = None
        self.progress_label = None
        self.photo = None
        
        # Pre-extract objects from first image
        self._load_current_image_objects()
        
    def _load_current_image_objects(self):
        """Load objects from current image"""
        if self.current_image_idx >= len(self.image_files):
            self.current_objects = []
            return
            
        image_filename = self.image_files[self.current_image_idx]
        image_path = os.path.join(self.images_dir, image_filename)
        
        # Run YOLO inference
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"⚠️ Image loading failed: {image_path}")
                self.current_objects = []
                return
            
            results = self.model.predict(
                source=img,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                save=False,
                verbose=False
            )
            
            # Extract detected objects
            self.current_objects = []
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
                            self.current_objects.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': conf,
                                'image': obj_img,
                                'index': i,
                                'labeled': False
                            })
            
            print(f"  📦 {image_filename}: {len(self.current_objects)} objects detected")
            
        except Exception as e:
            print(f"⚠️ Object detection failed ({image_filename}): {str(e)}")
            self.current_objects = []
    
    def setup_ui(self):
        """Set up UI"""
        self.root = tk.Tk()
        self.root.title("Manual Object Labeling - YOLO Active Learning")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Progress info label
        self.progress_label = ttk.Label(info_frame, text="", font=("Arial", 12, "bold"))
        self.progress_label.pack()
        
        # Detailed info label
        self.info_label = ttk.Label(info_frame, text="", font=("Arial", 10))
        self.info_label.pack()
        
        # Image display frame
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Canvas (with scroll support)
        canvas_frame = ttk.Frame(image_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='white', highlightthickness=1, highlightbackground='gray')
        scrollbar_v = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        scrollbar_h = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar_v.pack(side="right", fill="y")
        scrollbar_h.pack(side="bottom", fill="x")
        
        # Bottom button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Classification buttons (large and prominent)
        classify_frame = ttk.Frame(button_frame)
        classify_frame.pack(pady=(0, 10))
        
        # Class 0 button (green)
        self.class0_btn = tk.Button(classify_frame, text="✅ Class 0 (Keep)\nObjects to keep", 
                                   command=lambda: self.label_object(0),
                                   font=("Arial", 14, "bold"), 
                                   bg='#4CAF50', fg='white', 
                                   width=20, height=3,
                                   relief='raised', bd=3)
        self.class0_btn.pack(side=tk.LEFT, padx=10)
        
        # Class 1 button (red)
        self.class1_btn = tk.Button(classify_frame, text="❌ Class 1 (Filter)\nObjects to filter", 
                                   command=lambda: self.label_object(1),
                                   font=("Arial", 14, "bold"), 
                                   bg='#f44336', fg='white', 
                                   width=20, height=3,
                                   relief='raised', bd=3)
        self.class1_btn.pack(side=tk.LEFT, padx=10)
        
        # Navigation buttons
        nav_frame = ttk.Frame(button_frame)
        nav_frame.pack()
        
        ttk.Button(nav_frame, text="⬅️ Previous Object", 
                  command=self.prev_object).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="➡️ Next Object", 
                  command=self.next_object).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="⬆️ Previous Image", 
                  command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="⬇️ Next Image", 
                  command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        # Utility buttons
        util_frame = ttk.Frame(button_frame)
        util_frame.pack(pady=(10, 0))
        
        ttk.Button(util_frame, text="🔄 Refresh", 
                  command=self.refresh_display).pack(side=tk.LEFT, padx=5)
        ttk.Button(util_frame, text="📊 View Statistics", 
                  command=self.show_statistics).pack(side=tk.LEFT, padx=5)
        ttk.Button(util_frame, text="✅ Finish Labeling", 
                  command=self.finish_labeling).pack(side=tk.RIGHT, padx=5)
        
        # Keyboard shortcut bindings
        self.root.bind('<Key-1>', lambda e: self.label_object(0))
        self.root.bind('<Key-2>', lambda e: self.label_object(1))
        self.root.bind('<Left>', lambda e: self.prev_object())
        self.root.bind('<Right>', lambda e: self.next_object())
        self.root.bind('<Up>', lambda e: self.prev_image())
        self.root.bind('<Down>', lambda e: self.next_image())
        self.root.bind('<Escape>', lambda e: self.finish_labeling())
        
        # Set focus (to receive keyboard events)
        self.root.focus_set()
        
        print("✅ UI setup completed")
        print("📝 Shortcuts:")
        print("  - 1: Class 0 (Keep)")
        print("  - 2: Class 1 (Filter)")
        print("  - Arrow keys: Navigation")
        print("  - ESC: Finish")
    
    def update_display(self):
        """Update current image and object display"""
        if self.current_image_idx >= len(self.image_files):
            self.finish_labeling()
            return
        
        # Current image info
        image_filename = self.image_files[self.current_image_idx]
        image_path = os.path.join(self.images_dir, image_filename)
        
        # Update progress
        total_images = len(self.image_files)
        total_objects = len(self.current_objects)
        
        if total_objects == 0:
            # Move to next image if no objects
            self.next_image()
            return
        
        progress_text = (f"Image {self.current_image_idx + 1}/{total_images} | "
                        f"Object {self.current_object_idx + 1}/{total_objects} | "
                        f"Labeled: {self.total_labeled}")
        self.progress_label.config(text=progress_text)
        
        # Current object info
        current_obj = self.current_objects[self.current_object_idx]
        info_text = (f"File: {image_filename} | "
                    f"Confidence: {current_obj['confidence']:.3f} | "
                    f"Status: {'✅ Done' if current_obj['labeled'] else '⏳ Waiting'}")
        self.info_label.config(text=info_text)
        
        # Load and display image
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"⚠️ Image loading failed: {image_path}")
                return
            
            # Draw bounding boxes on image
            img_display = img.copy()
            
            for i, obj in enumerate(self.current_objects):
                bbox = obj['bbox']
                x1, y1, x2, y2 = bbox
                
                # Current object in green, others in gray
                if i == self.current_object_idx:
                    color = (0, 255, 0)  # Green
                    thickness = 3
                elif obj['labeled']:
                    color = (128, 128, 128)  # Gray (completed)
                    thickness = 1
                else:
                    color = (200, 200, 200)  # Light gray
                    thickness = 1
                
                cv2.rectangle(img_display, (x1, y1), (x2, y2), color, thickness)
                
                # Display object number
                label_text = f"#{i+1}"
                if obj['labeled']:
                    label_text += " ✓"
                
                cv2.putText(img_display, label_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display current object enlarged (top right)
            current_obj = self.current_objects[self.current_object_idx]
            obj_img = current_obj['image']
            
            # Resize object image (max 200x200)
            h, w = obj_img.shape[:2]
            max_size = 200
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                obj_img_resized = cv2.resize(obj_img, (new_w, new_h))
            else:
                obj_img_resized = obj_img
            
            # Composite object image on main image (top right)
            oh, ow = obj_img_resized.shape[:2]
            img_h, img_w = img_display.shape[:2]
            
            # Ensure margin
            margin = 10
            if img_w > ow + margin and img_h > oh + margin:
                # Draw background box
                cv2.rectangle(img_display, 
                             (img_w - ow - margin, margin), 
                             (img_w - margin, oh + margin + 30), 
                             (255, 255, 255), -1)
                cv2.rectangle(img_display, 
                             (img_w - ow - margin, margin), 
                             (img_w - margin, oh + margin + 30), 
                             (0, 0, 0), 2)
                
                # Composite object image
                img_display[margin:margin+oh, img_w-ow-margin:img_w-margin] = obj_img_resized
                
                # "Current Object" text
                cv2.putText(img_display, "Current Object", 
                           (img_w - ow - margin, margin + oh + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Convert OpenCV → PIL → PhotoImage
            img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Adjust to canvas size
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # Resize image
                img_w, img_h = img_pil.size
                scale = min(canvas_width/img_w, canvas_height/img_h) * 0.9
                
                if scale < 1:
                    new_w, new_h = int(img_w * scale), int(img_h * scale)
                    img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Create PhotoImage and display
            self.photo = ImageTk.PhotoImage(img_pil)
            
            # Clear canvas and display image
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, 
                                   image=self.photo, anchor=tk.CENTER)
            
            # Update scroll area
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            print(f"⚠️ Image display failed: {str(e)}")
    
    def save_object(self, obj_img, class_label, image_filename, obj_idx):
        """Save object image to corresponding class directory"""
        output_dir = self.class0_dir if class_label == 0 else self.class1_dir
        
        # Generate filename: original_image_name_object_index.jpg
        base_name = os.path.splitext(image_filename)[0]
        obj_filename = f"{base_name}_obj_{obj_idx:03d}.jpg"
        obj_path = os.path.join(output_dir, obj_filename)
        
        # Save image
        try:
            cv2.imwrite(obj_path, obj_img)
            print(f"  💾 Saved: {obj_filename} → Class {class_label}")
            return True
        except Exception as e:
            print(f"  ❌ Save failed: {obj_filename} - {str(e)}")
            return False
    
    def label_object(self, class_label):
        """Assign label to current object"""
        if not self.current_objects or self.current_object_idx >= len(self.current_objects):
            return
        
        current_obj = self.current_objects[self.current_object_idx]
        
        # Skip already labeled objects
        if current_obj['labeled']:
            self.next_object()
            return
        
        image_filename = self.image_files[self.current_image_idx]
        
        # Save object
        success = self.save_object(
            current_obj['image'], 
            class_label, 
            image_filename, 
            current_obj['index']
        )
        
        if success:
            # Mark as labeled
            current_obj['labeled'] = True
            self.total_labeled += 1
            
            # Auto move to next object
            self.next_object()
    
    def next_object(self):
        """Move to next object"""
        if not self.current_objects:
            self.next_image()
            return
        
        # Find next unlabeled object
        start_idx = self.current_object_idx
        while True:
            self.current_object_idx = (self.current_object_idx + 1) % len(self.current_objects)
            
            # If completed full circle, move to next image
            if self.current_object_idx == start_idx:
                self.next_image()
                break
            
            # Found unlabeled object
            if not self.current_objects[self.current_object_idx]['labeled']:
                break
        
        self.update_display()
    
    def prev_object(self):
        """Move to previous object"""
        if not self.current_objects:
            return
        
        self.current_object_idx = (self.current_object_idx - 1) % len(self.current_objects)
        self.update_display()
    
    def next_image(self):
        """Move to next image"""
        self.current_image_idx += 1
        self.current_object_idx = 0
        
        if self.current_image_idx >= len(self.image_files):
            self.finish_labeling()
            return
        
        # Load objects from new image
        self._load_current_image_objects()
        self.update_display()
    
    def prev_image(self):
        """Move to previous image"""
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.current_object_idx = 0
            self._load_current_image_objects()
            self.update_display()
    
    def refresh_display(self):
        """Refresh display"""
        self.update_display()
    
    def show_statistics(self):
        """Display labeling statistics"""
        class0_count = len(os.listdir(self.class0_dir))
        class1_count = len(os.listdir(self.class1_dir))
        total_labeled = class0_count + class1_count
        
        # Estimate total objects up to current image
        processed_images = self.current_image_idx
        avg_objects_per_image = self.total_labeled / max(1, processed_images) if processed_images > 0 else 0
        
        stats_message = f"""📊 Labeling Statistics
        
✅ Class 0 (Keep): {class0_count} items
❌ Class 1 (Filter): {class1_count} items
📊 Total labeled: {total_labeled} items

📸 Processed images: {processed_images}/{len(self.image_files)}
📦 Average objects/image: {avg_objects_per_image:.1f}

Progress: {processed_images/len(self.image_files)*100:.1f}%"""
        
        messagebox.showinfo("Labeling Statistics", stats_message)
    
    def finish_labeling(self):
        """Finish labeling"""
        class0_count = len(os.listdir(self.class0_dir))
        class1_count = len(os.listdir(self.class1_dir))
        total_count = class0_count + class1_count
        
        if total_count == 0:
            result = messagebox.askyesno("Warning", 
                                       "No objects have been labeled.\n"
                                       "Are you sure you want to exit?")
            if not result:
                return
        
        completion_message = f"""🎉 Labeling Completed!

📊 Final Results:
  ✅ Class 0 (Keep): {class0_count} items
  ❌ Class 1 (Filter): {class1_count} items
  📊 Total: {total_count} items

💾 Save Location:
  📁 {self.output_dir}

This data will be used for Classification model training."""
        
        messagebox.showinfo("Completed", completion_message)
        
        if self.root:
            self.root.quit()
            self.root.destroy()
    
    def run(self):
        """Run UI"""
        if not self.image_files:
            print("❌ No images to label.")
            return None
        
        print(f"🏷️ Starting Manual Labeling UI")
        print(f"  - Total images: {len(self.image_files)}")
        print(f"  - First image objects: {len(self.current_objects)}")
        
        # Set up and run UI
        self.setup_ui()
        
        # Initial display (after UI rendering)
        self.root.after(500, self.update_display)
        
        try:
            # Run main loop
            self.root.mainloop()
        except Exception as e:
            print(f"⚠️ Error during UI execution: {str(e)}")
        
        # Validate results
        class0_count = len(os.listdir(self.class0_dir))
        class1_count = len(os.listdir(self.class1_dir))
        
        if class0_count > 0 or class1_count > 0:
            print(f"✅ Labeling completed: Class 0={class0_count}, Class 1={class1_count}")
            return self.output_dir
        else:
            print("⚠️ No labeled data.")
            return None

if __name__ == "__main__":
    # Test execution
    print("🧪 Manual Labeling UI Test")
    
    # Test configuration
    ui = ManualLabelingUI(
        yolo_model_path="./results/01_initial_yolo/yolov8_100pct.pt",
        images_dir="./dataset/images",
        output_dir="./test_manual_labeling",
        conf_threshold=0.25,
        iou_threshold=0.5
    )
    
    # Run UI
    result = ui.run()
    
    if result:
        print(f"✅ Test completed: {result}")
    else:
        print("❌ Test failed")