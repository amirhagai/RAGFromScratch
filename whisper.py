import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from datasets import load_dataset, Audio, Dataset
# from hebrew_whisper import download_youtube_audio
import os 


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    generate_kwargs={"language": "he"}
)

# youtube_url = 'https://www.youtube.com/watch?v=fXNE8EBuvuc'

# # Download and extract audio
# audio_file = download_youtube_audio(youtube_url)


def load_chunk_paths(directory):
    chunk_paths = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            chunk_paths.append(file_path)
    return chunk_paths

# Example usage
chunk_paths = load_chunk_paths("/RAG/audio_chunks")
dataset = load_dataset('audiofolder', data_files=chunk_paths, split='train')


for i, example in enumerate(dataset):
    # Load the audio sample
    audio = example["audio"]

    # Perform speech recognition
    prediction = pipe(audio["array"], batch_size=8, return_timestamps=True)

    # Define the output text file path
    transcript_file = f'transcript_whisper_{i}.txt'

    # Save the transcript to the text file
    with open(transcript_file, 'w', encoding='utf-8') as f:
        f.write(prediction["text"])

    print(f"Transcript saved to {transcript_file}")

