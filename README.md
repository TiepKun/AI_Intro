# Taxi AI - Traffic Detecting Project
# Installation
## 1. Clone the repository & install:
```bash
- git clone git@github.com:ducquyen12312ew/AI_Intro.git
```
- Then, move to the directory:
```bash
cd “path of your project”
```
- Open terminal and run to install yolov5:
```bash
pip install torch opencv-python numpy ultralytics
```
# 2 How to use: 
Run simulation with webcam:
```bash
python main.py --debug
```
Run simulation with video:
```bash
Create a `output` foler in your project
python main.py --video path/of/your/video --output ./output/result.mp4 --debug
```
Detailed video analysis:
```bash
python video_analysis.py --input path/of/your/video --output ./output/analysis.mp4 --signs --debug
```

# Command Line Options
For main.py:
`--video`: Input video path (default: use webcam)
`--output`: Output video path (default: ./output/taxi_simulation.mp4)
`--traffic-only`: Use traffic light detector only (no sign detector)
`--debug`: Enable debug mode to show detailed information
`--skip`: Number of frames to skip between processing (default: 1, process 1/2 frames)
For video_analysis.py:
`--input`: Input video path (required)
`--output`: Output video path (default: ./output/analyzed_video.mp4)
`--signs`: Use sign detector with traffic lights
`--no-preview`: Do not show preview when processing
`--ratio`: Frames processing ratio (default: 2, process 1/2 frames)
`--debug`: Enable debug mode to display detailed information

