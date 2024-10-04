import argparse
import cv2
import logging
from datetime import datetime
import time
from model import Model

# Configure logging
logging.basicConfig(filename='detection_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def argument_parser():
    parser = argparse.ArgumentParser(description="Violence detection in images, videos, and webcam feed")
    parser.add_argument('--input-path', type=str, help='Path to your image or video file (leave empty for webcam)')
    args = parser.parse_args()
    return args

def log_label(label):
    """Log the detected label with the current timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f'{timestamp} - Detected label: {label}')

def process_image(model, image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    label = model.predict(image=image)['label']
    log_label(label)
    print(f'Predicted label for image: {label}')
    cv2.imshow('Violence Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(model, video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video from {video_path}")
        return

    last_log_time = time.time()
    
    while True:
        ret, frame = video.read()
        if not ret:
            break  # End of video
        
        label = model.predict(image=frame)['label']
        
        # Log the label every 200 ms
        current_time = time.time()
        if current_time - last_log_time >= 0.2:  # 200 ms
            log_label(label)
            last_log_time = current_time
            
        # Add label text on the frame
        cv2.putText(frame, f'Label: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Show the frame in the same window
        cv2.imshow('Violence Detection', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    video.release()
    cv2.destroyAllWindows()

def process_webcam(model):
    video = cv2.VideoCapture(0)  # Capture from the default webcam
    if not video.isOpened():
        print("Error: Could not access webcam.")
        return

    last_log_time = time.time()

    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        
        label = model.predict(image=frame)['label']
        
        # Log the label every 200 ms
        current_time = time.time()
        if current_time - last_log_time >= 0.2:  # 200 ms
            log_label(label)
            last_log_time = current_time
        
        # Add label text on the frame
        cv2.putText(frame, f'Label: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Show the frame in the same window
        cv2.imshow('Violence Detection', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = argument_parser()
    model = Model()
    input_path = args.input_path
    if input_path is None:
        process_webcam(model)
    elif input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        process_image(model, input_path)
    elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.flv')):
        process_video(model, input_path)
    else:
        print("Error: Unsupported file format. Please provide a valid image or video file, or leave empty for webcam.")
