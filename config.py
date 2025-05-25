import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    KAFKA_CONSUMER_GROUP = os.getenv('KAFKA_CONSUMER_GROUP', 'video-processing-group')
    KAFKA_INPUT_TOPIC = os.getenv('KAFKA_INPUT_TOPIC', 'video.processing')
    KAFKA_OUTPUT_TOPIC = os.getenv('KAFKA_OUTPUT_TOPIC', 'proctoring.results')
    
    # GPU Settings
    USE_CUDA = os.getenv('USE_CUDA', 'true').lower() == 'true'
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '1'))
    HALF_PRECISION = os.getenv('HALF_PRECISION', 'false').lower() == 'true'
    
    # Processing Settings
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
    FRAME_SKIP = int(os.getenv('FRAME_SKIP', '15'))
    
    # Thresholds
    GAZE_THRESHOLD = float(os.getenv('GAZE_THRESHOLD', '0.35'))
    DETECTION_CONFIDENCE = float(os.getenv('DETECTION_CONFIDENCE', '0.5'))