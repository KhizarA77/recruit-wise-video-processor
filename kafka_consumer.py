import json
import logging
from kafka import KafkaConsumer, KafkaProducer
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Dict, Any
from config import Config
from video_processor import VideoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessingConsumer:
    def __init__(self, config: Config):
        self.config = config
        self.video_processor = VideoProcessor(config)
        
        self.consumer = KafkaConsumer(
            config.KAFKA_INPUT_TOPIC,
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            group_id=config.KAFKA_CONSUMER_GROUP,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
        self.active_tasks = {}
        self.lock = threading.Lock()
        
        logger.info(f"VideoProcessingConsumer initialized with {config.MAX_WORKERS} workers")
    
    def process_video_message(self, message_value: Dict[str, Any]) -> None:
        """Process a single video message"""
        interview_id = message_value.get('interviewId')
        video_url = message_value.get('videoUrl')
        
        logger.info(f"Processing video for interview: {interview_id}")
        
        temp_video_path = None
        try:
            temp_video_path = self.video_processor.download_video(video_url)
            
            results = self.video_processor.process_video(temp_video_path, interview_id)
            
            self.producer.send(
                self.config.KAFKA_OUTPUT_TOPIC,
                value=results
            )
            self.producer.flush()
            
            logger.info(f"Results sent for interview: {interview_id}")
            
        except Exception as e:
            logger.error(f"Error processing video for interview {interview_id}: {e}")
            
            # Send error result
            error_result = {
                "interviewId": interview_id,
                "results": [{
                    "incident": f"Processing error: {str(e)}",
                    "timestamps": []
                }]
            }
            
            self.producer.send(
                self.config.KAFKA_OUTPUT_TOPIC,
                value=error_result
            )
            self.producer.flush()
            
        finally:
            if temp_video_path:
                self.video_processor.cleanup_temp_file(temp_video_path)
            
            with self.lock:
                if interview_id in self.active_tasks:
                    del self.active_tasks[interview_id]
    
    def start_consuming(self):
        """Start consuming messages from Kafka"""
        logger.info("Starting video processing consumer...")
        
        try:
            for message in self.consumer:
                try:
                    message_value = message.value
                    interview_id = message_value.get('interviewId')
                    
                    logger.info(f"Received message for interview: {interview_id}")
                    
                    with self.lock:
                        if interview_id in self.active_tasks:
                            logger.warning(f"Interview {interview_id} is already being processed, skipping...")
                            continue
                        
                        future = self.executor.submit(self.process_video_message, message_value)
                        self.active_tasks[interview_id] = future
                    
                    self.cleanup_completed_tasks()
                    
                except Exception as e:
                    logger.error(f"Error processing Kafka message: {e}")
                    
        except KeyboardInterrupt:
            logger.info("Shutting down consumer...")
        finally:
            self.shutdown()
    
    def cleanup_completed_tasks(self):
        """Clean up completed tasks from active_tasks"""
        with self.lock:
            completed_interviews = []
            for interview_id, future in self.active_tasks.items():
                if future.done():
                    completed_interviews.append(interview_id)
            
            for interview_id in completed_interviews:
                del self.active_tasks[interview_id]
    
    def shutdown(self):
        """Gracefully shutdown the consumer"""
        logger.info("Shutting down video processing consumer...")
        
        with self.lock:
            active_futures = list(self.active_tasks.values())
        
        for future in as_completed(active_futures, timeout=30):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in task during shutdown: {e}")
        
        self.executor.shutdown(wait=True)
        
        self.consumer.close()
        self.producer.close()
        
        logger.info("Consumer shutdown complete")