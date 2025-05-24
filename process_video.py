import cv2
import numpy as np
import librosa
import subprocess
import matplotlib.pyplot as plt

# Inspired by [Research Paper]'s multimodal framework, which uses CNNs for visual processing,
# Transformers for text, and RNNs for audio. This implementation uses simplified algorithms
# (mean pixel intensity, keyword scoring, spectral centroid) as proxies for neural network-based
# feature extraction, suitable for educational purposes.

# Step 1: Visual Feature Extraction (Proxy for CNN)
# Computes average brightness of frames, simulating visual energy detection.
def extract_visual_features(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return None

    frame_brightness = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        frame_brightness.append(brightness)

    cap.release()
    if frame_brightness:
        return np.mean(frame_brightness)
    return 0.0

# Step 2: Text Feature Extraction (Proxy for Transformer)
# Counts positive words to estimate description positivity, simulating sentiment analysis.
def extract_text_features(text):
    positive_words = ["intense", "fun", "energetic", "great", "awesome"]
    words = text.lower().split()
    positive_count = sum(1 for word in words if word in positive_words)
    return positive_count / max(len(words), 1)

# Step 3: Audio Feature Extraction (Proxy for RNN/Spectrogram Model)
# Computes spectral centroid to capture audio energy, simulating temporal feature extraction.
def extract_audio_features(video_path):
    temp_audio = "temp_audio.wav"
    try:
        subprocess.run(["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", temp_audio, "-y"], check=True)
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return 0.0

    try:
        y, sr = librosa.load(temp_audio)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        return np.mean(centroid)
    except Exception as e:
        print(f"Error processing audio: {e}")
        return 0.0

# Step 4: Late Fusion with User-Defined Weights (Simplified Attention)
# Combines normalized features using user weights, inspired by the research paper's late fusion.
def late_fusion(visual, text, audio, user_weights):
    visual_norm = min(visual / 255.0, 1.0)  # Normalize brightness (0-255)
    text_norm = text  # Already 0-1
    audio_norm = min(audio / 5000.0, 1.0)  # Normalize centroid (0-5000 Hz)
    return (user_weights[0] * visual_norm +
            user_weights[1] * text_norm +
            user_weights[2] * audio_norm)

# Step 5: Visualization
# Plots normalized features and final score, showing modality contributions.
def visualize_system(visual, text, audio, final_score, user_weights):
    modalities = ['Visual', 'Text', 'Audio']
    features = [min(visual / 255.0, 1.0), text, min(audio / 5000.0, 1.0)]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(modalities, features, alpha=0.5, label='Normalized Features')
    plt.axhline(y=final_score, color='r', linestyle='--', label=f'Final Score: {final_score:.2f}')
    plt.xlabel('Modalities')
    plt.ylabel('Feature Values (Normalized)')
    plt.title(f'Multimodal Analysis for intense_workout.mp4\nWeights: Visual={user_weights[0]:.2f}, Text={user_weights[1]:.2f}, Audio={user_weights[2]:.2f}')
    for bar, feature in zip(bars, features):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{feature:.2f}', ha='center', va='bottom')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Step 6: Get User Preferences
# Collects and normalizes user weights, simulating personalized fusion.
def get_user_weights():
    print("Enter weights for each modality (0 to 1, sum should be close to 1):")
    while True:
        try:
            visual_weight = float(input("Weight for Visual (e.g., 0.3): "))
            text_weight = float(input("Weight for Text (e.g., 0.3): "))
            audio_weight = float(input("Weight for Audio (e.g., 0.4): "))
            weights = [visual_weight, text_weight, audio_weight]
            total = sum(weights)
            if total <= 0 or any(w < 0 for w in weights):
                print("Weights must be positive and sum to a positive value. Try again.")
                continue
            weights = [w / total for w in weights]
            print(f"Normalized Weights: Visual={weights[0]:.2f}, Text={weights[1]:.2f}, Audio={weights[2]:.2f}")
            return weights
        except ValueError:
            print("Please enter valid numbers. Try again.")

# Main execution
if __name__ == "__main__":
    video_path = "intense_workout.mp4"
    text_description = "This is an intense and energetic workout video"

    # Get user preferences
    user_weights = get_user_weights()

    # Extract features
    visual_feature = extract_visual_features(video_path)
    text_feature = extract_text_features(text_description)
    audio_feature = extract_audio_features(video_path)

    if visual_feature is not None:
        # Compute final score
        final_score = late_fusion(visual_feature, text_feature, audio_feature, user_weights)

        # Print results
        print(f"\nResults for {video_path}:")
        print(f"Visual Feature (Average Brightness): {visual_feature:.2f}")
        print(f"Text Feature (Positivity Score): {text_feature:.2f}")
        print(f"Audio Feature (Spectral Centroid): {audio_feature:.2f}")
        print(f"Final Recommendation Score: {final_score:.2f}")

        # Visualize
        visualize_system(visual_feature, text_feature, audio_feature, final_score, user_weights)
        