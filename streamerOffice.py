import logging
import queue
import time
import cv2
import numpy as np
import os
import sys
from app import process_frame, frame_queue, stats_lock, tracker
from collections import deque
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced RTSP configuration
RTSP_URL = "rtsp://admin:Mmmycash@6699@mycash.ddns.net:56100/" + \
           "?tcp&buffer_size=262144&rtsp_transport=tcp" + \
           "&timeout=5000000&analyzeduration=5000000" + \
           "&probesize=5000000&flush_packets=1" + \
           "&reorder_queue_size=5000&max_delay=1000000"

#testing with Video file:
VIDEO_FILE = "testVid.mp4"

RECONNECT_DELAY = 3  # seconds
MAX_RETRIES = 10
FRAME_SKIP = 6  # Process every 6th frame
FRAME_TIMEOUT = 15  # seconds without frames before reconnecting
MAX_CONSECUTIVE_ERRORS = 100

class VideoStreamer:
    def __init__(self):
        self.cap = None
        self.last_frame_time = time.time()
        self.consecutive_errors = 0
        self.frame_count = 0
        self.retries = 0

    def get_video_capture(self):
        # Configure FFmpeg options
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = \
            "rtsp_transport;tcp|buffer_size;262144|analyzeduration;5000000"
        
        # Try multiple backends
        backends = [
            (cv2.CAP_FFMPEG, "FFmpeg"),
            (cv2.CAP_GSTREAMER, "GStreamer"),
            (cv2.CAP_ANY, "Any")
        ]
        
        for backend, name in backends:
            #cap = cv2.VideoCapture(RTSP_URL, backend)
            cap = cv2.VideoCapture(VIDEO_FILE)
            if cap.isOpened():
                #logger.info(f"Connected using {name} backend")
                logger.info(f"Could not open video file: {VIDEO_FILE}")
                self.configure_capture(cap)
                return cap
        
        raise ConnectionError("All backends failed to connect")

    def configure_capture(self, cap):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def process_stream(self):
        while self.retries < MAX_RETRIES:
            try:
                if self.cap is None or not self.cap.isOpened():
                    logger.info(f"Connection attempt {self.retries + 1}/{MAX_RETRIES}")
                    self.cap = self.get_video_capture()
                    self.retries = 0
                    self.consecutive_errors = 0
                    logger.info("Stream connected successfully")

                while True:
                    current_time = time.time()
                    if current_time - self.last_frame_time > FRAME_TIMEOUT:
                        logger.warning(f"No frames for {FRAME_TIMEOUT}s, reconnecting...")
                        break

                    ret, frame = self.cap.read()
                    if not ret:
                        self.consecutive_errors += 1
                        if self.consecutive_errors > MAX_CONSECUTIVE_ERRORS:
                            logger.warning("Max consecutive errors reached, reconnecting...")
                            break
                        time.sleep(0.1)
                        continue

                    # Validate frame
                    if frame is None or frame.size == 0:
                        logger.debug("Empty frame received")
                        continue

                    # Reset error counters
                    self.consecutive_errors = 0
                    self.last_frame_time = current_time

                    # Process frame
                    if self.frame_count % FRAME_SKIP == 0 and not frame_queue.full():
                        try:
                            frame_queue.put(frame.copy(), timeout=0.5)
                        except queue.Full:
                            logger.debug("Frame queue full, skipping")

                    self.frame_count += 1
                    time.sleep(0.033)  # ~30fps

            except (cv2.error, ConnectionError) as e:
                logger.error(f"Stream error: {str(e)}")
                self.handle_error()
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                self.handle_error()

        logger.error("Max retries reached, exiting")

    def handle_error(self):
        self.retries += 1
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        time.sleep(RECONNECT_DELAY)

def process_frames():
    frame_count = 0
    last_update_time = time.time()

    while True:
        try:
            frame = frame_queue.get(timeout=1.0)

            # Only process every 6th frame (adjust as needed)
            if frame_count % 6 == 0:
                process_frame(frame, frame_count, 6)

                # Update Firebase every 30 frames (~1 second at 30fps)
                if frame_count % 30 == 0:
                    tracker.update_statistics()

            frame_count += 1



        except queue.Empty:
            logger.warning("Frame queue empty, waiting...")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # Set process priority
    try:
        os.nice(10)
    except:
        logger.warning("Could not adjust process priority")

    # Initialize and start streamer
    streamer = VideoStreamer()
    stream_thread = threading.Thread(target=streamer.process_stream, daemon=True)
    processor_thread = threading.Thread(target=process_frames, daemon=True)
    
    stream_thread.start()
    processor_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        if streamer.cap is not None:
            streamer.cap.release()
