import cv2
import numpy as np
import librosa
import subprocess
import matplotlib.pyplot as plt

# Step 1: Read video and extract visual features
def extract_visual_features(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
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

# Step 2: Simulate text feature extraction
def extract_text_features(text):
    positive_words = ["intense", "fun", "energetic", "great"]
    words = text.lower().split()
    positive_count = sum(1 for word in words if word in positive_words)
    return positive_count / max(len(words), 1)

# Step 3: Extract audio features
def extract_audio_features(video_path):
    temp_audio = "temp_audio.wav"
    try:
        subprocess.run(["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", temp_audio, "-y"], check=True)
    except Exception as e:
        print("Error extracting audio:", e)
        return 0.0

    try:
        y, sr = librosa.load(temp_audio)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        return np.mean(centroid)
    except Exception as e:
        print("Error processing audio:", e)
        return 0.0

# Step 4: Late fusion with attention
def late_fusion(visual, text, audio):
    visual_norm = min(visual / 255.0, 1.0)
    text_norm = text
    audio_norm = min(audio / 5000.0, 1.0)
    weights = [0.33, 0.33, 0.34]
    return weights[0] * visual_norm + weights[1] * text_norm + weights[2] * audio_norm

# Step 5: Visualize the system
def visualize_system(visual, text, audio, final_score):
    modalities = ['Visual', 'Text', 'Audio']
    features = [min(visual / 255.0, 1.0), text, min(audio / 5000.0, 1.0)]
    plt.bar(modalities, features, alpha=0.5, label='Normalized Features')
    plt.axhline(y=final_score, color='r', linestyle='--', label=f'Final Score: {final_score:.2f}')
    plt.xlabel('Modalities')
    plt.ylabel('Feature Values (Normalized)')
    plt.title('Multimodal System: Feature Contributions')
    plt.legend()
    plt.show()

# Main execution
if __name__ == "__main__":
    video_path = "intense_workout.mp4"
    text_description = "This is an intense and energetic workout video"

    visual_feature = extract_visual_features(video_path)
    text_feature = extract_text_features(text_description)
    audio_feature = extract_audio_features(video_path)

    if visual_feature is not None:
        final_score = late_fusion(visual_feature, text_feature, audio_feature)
        print(f"Visual Feature (Average Brightness): {visual_feature:.2f}")
        print(f"Text Feature (Positivity Score): {text_feature:.2f}")
        print(f"Audio Feature (Spectral Centroid): {audio_feature:.2f}")
        print(f"Final Recommendation Score: {final_score:.2f}")
        visualize_system(visual_feature, text_feature, audio_feature, final_score)