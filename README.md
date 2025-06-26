# Real-Time AI Object Detection with YOLOv8 and Webcam
This project provides a ready-to-use Python script for performing real-time object detection on a live webcam feed using the powerful YOLOv8 model. It is built with OpenCV for video handling and PyTorch for deep learning, automatically leveraging a GPU if one is available.

# Features
Live Webcam Detection: Captures your webcam feed and performs detection in real-time.

State-of-the-Art Model: Uses a pre-trained YOLOv8n model from Ultralytics, which is fast and accurate.

Automatic Model Download: The required YOLOv8 model is downloaded automatically on the first run.

GPU Acceleration: Automatically detects and uses an available NVIDIA GPU (with CUDA) for a significant performance boost. Falls back to CPU otherwise.

Annotated Output: Draws bounding boxes and labels for detected objects directly onto the video stream.

Save Results: Automatically saves the annotated frames into an outputs/ directory.

Easy to Use: Simply run a single Python script to start the application.

Clean and Modular Code: The code is well-commented and structured for readability and easy modification.

# Project Structure
yolo-webcam-detection/

├── live_detection.py     # Main script for webcam detection

├── requirements.txt      # List of Python dependencies

├── README.md             # This file

└── outputs/              # Directory where detected frames are saved (created automatically)

# Setup Instructions
# 1. Clone or Download the Project
First, get the project files onto your local machine.

# 2. Create a Python Virtual Environment (Recommended)
It is highly recommended to use a virtual environment to avoid conflicts with other Python projects.

# Create a new virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install Dependencies
Install all the required libraries using the requirements.txt file.

pip install -r requirements.txt

Note on GPU Support: The requirements.txt file installs the standard PyTorch version. If you have an NVIDIA GPU, ensure your drivers and CUDA toolkit are installed correctly. PyTorch will automatically use them. For specific CUDA versions, you might need a custom PyTorch installation command from the official PyTorch website.

How to Run the Application
Once the setup is complete, you can run the object detection script from your terminal:

python live_detection.py

A window will pop up showing your live webcam feed.

Detected objects will be highlighted with bounding boxes and labels.

The annotated frames will be saved in the outputs/ folder.

To stop the application, press the q key while the webcam window is active.

Model Information
This project uses YOLOv8n ("nano"), the smallest and fastest version of the YOLOv8 family, making it ideal for real-time applications even on less powerful hardware. It is pre-trained on the COCO dataset, which can identify 80 common object classes like:

person, car, bicycle, motorcycle

dog, cat, bird

bottle, chair, laptop, cell phone

and many more.
