# Workout Video Processor

A Python-based multimodal system to process workout videos, extracting visual, text, and audio features to generate a personalized recommendation score. This project is inspired by [Research Paper]'s multimodal framework, which uses Convolutional Neural Networks (CNNs) for visual processing, Transformers for text, and Recurrent Neural Networks (RNNs) for audio.

## Features
- Extracts visual features (average brightness) as a proxy for CNN-based processing.
- Extracts text features (positivity score) as a proxy for Transformer-based sentiment analysis.
- Extracts audio features (spectral centroid) as a proxy for RNN-based audio analysis.
- Combines features using late fusion with user-defined weights, simulating attention-like personalization.
- Visualizes modality contributions with a bar plot.

## Requirements
- Python 3.8+
- Libraries: `opencv-python`, `librosa`, `numpy`, `matplotlib`
- Tool: `ffmpeg` (install via `brew install ffmpeg`)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/lakshminpacha/WorkoutVideoProcessor.git
   cd WorkoutVideoProcessor