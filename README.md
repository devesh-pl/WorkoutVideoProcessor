# Workout Video Processor

A Python-based multimodal system to process workout videos, extracting visual, text, and audio features to generate a recommendation score.

## Requirements
- Python 3.8+
- Libraries: `opencv-python`, `librosa`, `numpy`, `matplotlib`
- Tool: `ffmpeg` (install via `brew install ffmpeg`)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/lakshminpacha/WorkoutVideoProcessor.git
   cd WorkoutVideoProcessor
   pip3 install opencv-python librosa numpy matplotlib
brew install ffmpeg
python3 process_video.py
Output
Visual Feature: Average brightness of video frames.
Text Feature: Positivity score from the description.
Audio Feature: Spectral centroid of the audio.
Final Score: Combined recommendation score.
Plot: Bar chart of normalized features and final score.
