from pydub import AudioSegment
import math
import os

def split_audio(file_path, chunk_length_ms):
    # Load the audio file
    audio = AudioSegment.from_file(file_path)
    # Calculate the number of chunks
    total_length_ms = len(audio)
    num_chunks = math.ceil(total_length_ms / chunk_length_ms)
    # Create output directory
    output_dir = "audio_chunks"
    os.makedirs(output_dir, exist_ok=True)
    # Split and export chunks
    for i in range(num_chunks):
        start_time = i * chunk_length_ms
        end_time = min((i + 1) * chunk_length_ms, total_length_ms)
        chunk = audio[start_time:end_time]
        chunk.export(os.path.join(output_dir, f"chunk_{i + 1}.wav"), format="wav")
        print(f"Exported chunk_{i + 1}.wav")

if __name__ == "__main__":
    file_path = "/RAG/audio.wav"  # Replace with your audio file path
    chunk_length_ms = 60 * 1000  # 30 seconds
    split_audio(file_path, chunk_length_ms)
