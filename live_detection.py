import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import os
import time

def load_model(device):
    """
    Loads a pre-trained YOLOv8 model.

    Args:
        device (torch.device): The device (CPU or CUDA) to load the model onto.

    Returns:
        ultralytics.YOLO: The loaded YOLOv8 model.
    """
    print("Loading YOLOv8 model...")
    # Load the YOLOv8n (nano) model, which is small and fast.
    # The model is automatically downloaded on the first run.
    try:
        model = YOLO('yolov8n.pt')
        model.to(device)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check your internet connection and ultralytics installation.")
        exit()

def run_webcam_detection(model, device):
    """
    Runs real-time object detection on a live webcam stream.

    Args:
        model (ultralytics.YOLO): The loaded YOLOv8 model.
        device (torch.device): The device to run inference on.
    """
    # --- 1. SETUP ---
    # Open a connection to the default webcam (usually index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create the output directory if it doesn't exist
    output_dir = Path('outputs')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting webcam detection... Press 'q' to quit.")

    frame_count = 0
    start_time = time.time()

    # --- 2. REAL-TIME DETECTION LOOP ---
    while True:
        # Read a frame from the webcam
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame. Exiting.")
            break

        # --- 3. INFERENCE ---
        # Perform inference on the current frame
        # The model returns a list of Results objects
        results = model(frame, device=device, verbose=False)

        # --- 4. VISUALIZATION AND SAVING ---
        # The results object contains all detection data.
        # .plot() is a helper function to draw boxes and labels on the frame.
        annotated_frame = results[0].plot()

        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        # Display the annotated frame in a window
        cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

        # Save the annotated frame to the outputs folder
        # We can save every frame or based on a condition
        output_path = output_dir / f"frame_{frame_count:04d}.jpg"
        cv2.imwrite(str(output_path), annotated_frame)

        # --- 5. EXIT CONDITION ---
        # Check for the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 6. CLEANUP ---
    # Release the webcam resource
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    print("Webcam detection stopped.")

def main():
    """
    Main function to set up the device and run the detection pipeline.
    """
    # Determine the device to use for inference (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load the model
    model = load_model(device)

    # Run the webcam detection
    run_webcam_detection(model, device)

if __name__ == '__main__':
    main()
