import logging
from config import Config
from kafka_consumer import VideoProcessingConsumer

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = Config()
    consumer = VideoProcessingConsumer(config)
    
    try:
        consumer.start_consuming()
    except Exception as e:
        logging.error(f"Error in main: {e}")
    finally:
        consumer.shutdown()

if __name__ == "__main__":
    main()