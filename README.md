# Python Video Player with Mouse Tracking

A Python-based video player application with mouse tracking capabilities using YOLOv8 object detection. The application allows you to play videos, track mouse movements in the video, and includes a timer feature.

## Features

- Video playback controls (play/pause)
- Mouse tracking using YOLOv8 object detection
- Timer functionality
- Seek bar for video navigation
- Real-time visualization of detected mice

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for better performance)

## Installation

1. Clone or download this repository to your local machine.

2. Create and activate a virtual environment (recommended):

   Windows:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the YOLOv8 model:
   The `yolov8n.pt` model file should be in the root directory of the project. If it's not present, it will be automatically downloaded when running the application for the first time.

## Running the Application

1. Make sure your virtual environment is activated (if you created one).

2. Run the video player:
   ```bash
   python video_player.py
   ```

## Usage

1. Click the "Open Video" button to select a video file.

2. Use the play/pause button or spacebar to control video playback.

3. Enable mouse tracking by checking the "Enable Mouse Tracking" checkbox.
   - The application will display bounding boxes around detected mice
   - Mouse positions will be tracked and visualized

4. Use the timer feature:
   - Press the "Start Timer" button or the "1" key to start/stop the timer
   - Click "Reset Timer" to reset the timer to zero

5. Use the seek bar to navigate through the video.

## Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are correctly installed
2. Verify that the YOLO model file is present in the project directory
3. Check that your system meets the minimum requirements
4. Make sure you have sufficient GPU memory if using CUDA

## License

This project is open source and available under the MIT License.