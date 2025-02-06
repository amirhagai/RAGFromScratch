import os
from pytube import YouTube
from pydub import AudioSegment
import torch
from transformers import pipeline
from datasets import load_dataset, Audio, Dataset
from transformers import pipeline, AutoProcessor


# def download_youtube_audio(url, output_path='audio.wav'):
#     # Download the YouTube video
#     yt = YouTube(url)
#     audio_stream = yt.streams.filter(only_audio=True).first()
#     downloaded_file = audio_stream.download(filename='temp_audio')

#     # Convert the downloaded file to WAV format
#     audio = AudioSegment.from_file(downloaded_file)
#     audio.export(output_path, format='wav')

#     # Remove the temporary file
#     os.remove(downloaded_file)

#     return output_path

from yt_dlp import YoutubeDL

def download_youtube_audio(url, output_path='audio.wav'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Rename the downloaded file to the desired output path
    os.rename('temp_audio.wav', output_path)

    return output_path


device = "cuda:0" if torch.cuda.is_available() else "cpu"


cache_directory = "/RAG/hebrew_whisper"

# processor = AutoProcessor.from_pretrained("ivrit-ai/whisper-v2-d3-e3", cache_dir=cache_directory)
# processor = AutoProcessor.from_pretrained("ivrit-ai/whisper-v2-d3-e3", cache_dir=cache_directory)
# forced_decoder_ids = processor.get_decoder_prompt_ids(language="he", task="transcribe")
# pipe = pipeline(
#     "automatic-speech-recognition",
#     model="ivrit-ai/whisper-v2-d3-e3",
#     chunk_length_s=30,
#     device=device,
#     model_kwargs={
#         "cache_dir": cache_directory,
#         "forced_decoder_ids": forced_decoder_ids
#     }
# )


pipe = pipeline(
    "automatic-speech-recognition",
    model="ivrit-ai/whisper-v2-d3-e3",
    chunk_length_s=30,
    device=device,
    model_kwargs={"cache_dir": cache_directory},
    generate_kwargs={"language": "he"}  # Specify Hebrew language
    # task="transcribe"  # Ensure the task is set to transcribe
)

# URL of the YouTube video
youtube_url = 'https://www.youtube.com/watch?v=fXNE8EBuvuc'

# Download and extract audio
audio_file = download_youtube_audio(youtube_url)


# Create a dataset from the audio file
dataset = Dataset.from_dict({"audio": [audio_file]})
dataset = dataset.cast_column("audio", Audio())





# Retrieve the audio sample
sample = dataset[0]["audio"]

# Perform speech recognition
prediction = pipe(sample["array"], batch_size=8)["text"]

# Define the output text file path
transcript_file = 'transcript.txt'

# Save the transcript to the text file
with open(transcript_file, 'w', encoding='utf-8') as f:
    f.write(prediction)

print(f"Transcript saved to {transcript_file}")



# # Load the audio file into a dataset
# ds = load_dataset("audio", data_files=audio_file, split="train")
# sample = ds[0]["audio"]

# # Perform speech recognition
# prediction = pipe(sample["array"], batch_size=8)["text"]

# # Define the output text file path
# transcript_file = 'transcript.txt'

# # Save the transcript to the text file
# with open(transcript_file, 'w', encoding='utf-8') as f:
#     f.write(prediction)

# print(f"Transcript saved to {transcript_file}")
