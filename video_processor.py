import cv2
import numpy as np
import torch
import time
import os
import requests
import tempfile
from datetime import datetime, timedelta
from ultralytics import YOLO
import mediapipe as mp
import logging
from typing import List, Dict, Any, Tuple
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.device = 'cuda' if config.USE_CUDA and torch.cuda.is_available() else 'cpu'
        
        if self.device == 'cuda':
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            logger.info(f"CUDA version: {torch.version.cuda}")
            
            torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        logger.info(f"Initializing VideoProcessor on {self.device}")
        
        self.object_detector = YOLO("yolov8x.pt")
        self.object_detector.to(self.device)
        
        # Warm up the model on GPU
        if self.device == 'cuda':
            self._warmup_gpu()
        
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.target_classes = {
            0: "person",
            67: "cell phone", 
            73: "book",
            84: "book"
        }
        
        logger.info("VideoProcessor initialized successfully")
    
    def _warmup_gpu(self):
        """Warm up GPU with dummy inference"""
        logger.info("Warming up GPU...")
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        for _ in range(3):
            self.object_detector(dummy_image, verbose=False, device=self.device)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        logger.info("GPU warmup completed")
    
    def download_video(self, video_url: str) -> str:
        """Download video from S3 URL to temporary file"""
        try:
            logger.info(f"Downloading video from: {video_url}")
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            
            temp_file.close()
            logger.info(f"Video downloaded to: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            raise
    
    def get_gaze_direction(self, image_width: int, image_height: int, 
                          face_landmarks_normalized: List, gaze_threshold: float = 0.35) -> Tuple[str, bool, float]:
        """Analyze face landmarks to determine gaze direction"""
        sum_gaze_ratios = 0.0
        valid_eyes_count = 0

        left_iris_indices = list(range(473, 478))
        right_iris_indices = list(range(468, 473))

        left_eye_outer_corner_idx = 33
        left_eye_inner_corner_idx = 133
        right_eye_outer_corner_idx = 263
        right_eye_inner_corner_idx = 362

        try:
            left_iris_coords_x = [face_landmarks_normalized[i].x * image_width for i in left_iris_indices]
            left_pupil_x = np.mean(left_iris_coords_x)
            left_outer_x = face_landmarks_normalized[left_eye_outer_corner_idx].x * image_width
            left_inner_x = face_landmarks_normalized[left_eye_inner_corner_idx].x * image_width
            
            actual_left_eye_outer_x = min(left_outer_x, left_inner_x)
            actual_left_eye_inner_x = max(left_outer_x, left_inner_x)
            eye_width_left = abs(actual_left_eye_inner_x - actual_left_eye_outer_x)

            if eye_width_left > 5:
                eye_center_x_left = (actual_left_eye_outer_x + actual_left_eye_inner_x) / 2
                gaze_ratio_left = (left_pupil_x - eye_center_x_left) / (eye_width_left / 2 + 1e-6)
                sum_gaze_ratios += gaze_ratio_left
                valid_eyes_count += 1
        except Exception:
            pass

        try:
            right_iris_coords_x = [face_landmarks_normalized[i].x * image_width for i in right_iris_indices]
            right_pupil_x = np.mean(right_iris_coords_x)
            right_outer_x = face_landmarks_normalized[right_eye_outer_corner_idx].x * image_width
            right_inner_x = face_landmarks_normalized[right_eye_inner_corner_idx].x * image_width

            actual_right_eye_inner_x = min(right_outer_x, right_inner_x)
            actual_right_eye_outer_x = max(right_outer_x, right_inner_x)
            eye_width_right = abs(actual_right_eye_outer_x - actual_right_eye_inner_x)

            if eye_width_right > 5:
                eye_center_x_right = (actual_right_eye_inner_x + actual_right_eye_outer_x) / 2
                gaze_ratio_right = (right_pupil_x - eye_center_x_right) / (eye_width_right / 2 + 1e-6)
                sum_gaze_ratios += gaze_ratio_right
                valid_eyes_count += 1
        except Exception:
            pass

        if valid_eyes_count > 0:
            final_gaze_ratio = sum_gaze_ratios / valid_eyes_count
            if final_gaze_ratio < -gaze_threshold:
                return "LEFT", True, final_gaze_ratio
            elif final_gaze_ratio > gaze_threshold:
                return "RIGHT", True, final_gaze_ratio
            else:
                return "CENTER", False, final_gaze_ratio

        return "N/A", False, 0.0
    
    def process_video(self, video_path: str, interview_id: str) -> Dict[str, Any]:
        """Process video for cheating detection with GPU optimizations"""
        logger.info(f"Processing video: {video_path} for interview: {interview_id}")
        
        violations = {
            "cell_phone": [],
            "additional_person": [],
            "book": [],
            "person_missing": [],
            "looking_away": []
        }
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cuda_params = {
            'verbose': False,
            'device': self.device,
            'conf': self.config.DETECTION_CONFIDENCE,
            'iou': 0.45,
            'batch': self.config.BATCH_SIZE,
            'half': self.config.HALF_PRECISION and self.device == 'cuda',
            'imgsz': 640,  
            'max_det': 300  
        }
        
        frame_count = 0
        start_time = time.time()
        batch_frames = []
        batch_frame_indices = []
        
        with self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.config.FRAME_SKIP != 0:
                    frame_count += 1
                    continue
                
                timestamp = frame_count / fps
                
                if self.device == 'cuda' and self.config.BATCH_SIZE > 1:
                    batch_frames.append(frame)
                    batch_frame_indices.append((frame_count, timestamp))
                    
                    if len(batch_frames) >= self.config.BATCH_SIZE:
                        self._process_batch(batch_frames, batch_frame_indices, violations, 
                                          cuda_params, face_mesh)
                        batch_frames = []
                        batch_frame_indices = []
                else:
                    self._process_single_frame(frame, timestamp, violations, 
                                             cuda_params, face_mesh)
                
                frame_count += 1
                
                if frame_count % (fps * 10) == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / progress) * (100 - progress)
                    logger.info(f"Processing progress: {progress:.1f}% | ETA: {eta:.1f}s")
                
                if self.device == 'cuda' and frame_count % (fps * 120) == 0:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            if batch_frames:
                self._process_batch(batch_frames, batch_frame_indices, violations, 
                                  cuda_params, face_mesh)
        
        cap.release()
        processing_time = time.time() - start_time
        
        logger.info(f"Video processing completed in {processing_time:.2f} seconds")
        logger.info(f"Average FPS: {frame_count / processing_time:.2f}")
        
        if self.device == 'cuda':
            logger.info(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
            torch.cuda.empty_cache()
        
        return self.format_results(violations, interview_id)
    
    def _process_batch(self, frames: List[np.ndarray], frame_indices: List[Tuple[int, float]], 
                      violations: Dict[str, List[float]], cuda_params: Dict, face_mesh):
        
        object_results = self.object_detector(frames, **cuda_params)
        
        for i, (frame_count, timestamp) in enumerate(frame_indices):
            person_violations = self.detect_person_count_violations(object_results[i:i+1])
            
            if person_violations["person_missing"]:
                violations["person_missing"].append(timestamp)
            
            if person_violations["additional_person"]:
                violations["additional_person"].append(timestamp)
            
            for detection in object_results[i].boxes.data.tolist():
                class_id = int(detection[5])
                confidence = detection[4]
                
                if class_id in self.target_classes and confidence > self.config.DETECTION_CONFIDENCE:
                    object_type = self.target_classes[class_id]
                    
                    if object_type == "cell phone":
                        violations["cell_phone"].append(timestamp)
                    elif object_type == "book":
                        violations["book"].append(timestamp)
            
            self._process_gaze(frames[i], timestamp, violations, face_mesh)
    
    def _process_single_frame(self, frame: np.ndarray, timestamp: float, 
                            violations: Dict[str, List[float]], cuda_params: Dict, face_mesh):
        object_results = self.object_detector(frame, **cuda_params)
        
        person_violations = self.detect_person_count_violations(object_results)
        
        if person_violations["person_missing"]:
            violations["person_missing"].append(timestamp)
        
        if person_violations["additional_person"]:
            violations["additional_person"].append(timestamp)
        
        for detection in object_results[0].boxes.data.tolist():
            class_id = int(detection[5])
            confidence = detection[4]
            
            if class_id in self.target_classes and confidence > self.config.DETECTION_CONFIDENCE:
                object_type = self.target_classes[class_id]
                
                if object_type == "cell phone":
                    violations["cell_phone"].append(timestamp)
                elif object_type == "book":
                    violations["book"].append(timestamp)
        
        self._process_gaze(frame, timestamp, violations, face_mesh)
    
    def _process_gaze(self, frame: np.ndarray, timestamp: float, 
                     violations: Dict[str, List[float]], face_mesh):
        """Process gaze detection for a frame"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        gaze_results = face_mesh.process(image_rgb)
        
        if gaze_results.multi_face_landmarks:
            for face_landmarks_obj in gaze_results.multi_face_landmarks:
                face_landmarks_list = face_landmarks_obj.landmark
                img_h, img_w = frame.shape[:2]
                
                gaze_direction, is_looking_away, gaze_ratio = self.get_gaze_direction(
                    img_w, img_h, face_landmarks_list, self.config.GAZE_THRESHOLD
                )
                
                if is_looking_away:
                    violations["looking_away"].append(timestamp)
    
    def detect_person_count_violations(self, results) -> Dict[str, bool]:
        """Detect person count violations"""
        if len(results[0].boxes) == 0:
            return {"person_missing": True, "additional_person": False}
        
        persons = [detection for detection in results[0].boxes.data.tolist() 
                  if detection[5] == 0] 
        
        person_count = len(persons)
        
        return {
            "person_missing": person_count == 0,
            "additional_person": person_count > 1
        }
    
    def consolidate_violations(self, timestamps: List[float], time_threshold: float = 3.0) -> List[float]:
        """Consolidate nearby violation timestamps"""
        if not timestamps:
            return []
        
        timestamps = sorted(timestamps)
        consolidated = [timestamps[0]]
        
        for timestamp in timestamps[1:]:
            if timestamp - consolidated[-1] > time_threshold:
                consolidated.append(timestamp)
        
        return consolidated
    
    def format_results(self, violations: Dict[str, List[float]], interview_id: str) -> Dict[str, Any]:
        """Format results for Kafka message"""
        results = []
        
        violation_messages = {
            "cell_phone": "Cell phone detected",
            "additional_person": "Additional person detected", 
            "book": "Unauthorized material detected",
            "person_missing": "Person absent from frame",
            "looking_away": "Person looking away"
        }
        
        for violation_type, timestamps in violations.items():
            if timestamps:
                consolidated_timestamps = self.consolidate_violations(timestamps)
                if consolidated_timestamps:
                    results.append({
                        "incident": violation_messages[violation_type],
                        "timestamps": consolidated_timestamps
                    })
        
        return {
            "interviewId": interview_id,
            "results": results
        }
    
    def cleanup_temp_file(self, file_path: str):
        """Clean up temporary video file"""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up file {file_path}: {e}")