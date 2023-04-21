import soundfile as sf
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import gradio as gr
import os


api_token = "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"
model_name = "indonesian-nlp/wav2vec2-indonesian-javanese-sundanese"
cache_dir = "/usr/src/app/model_cache"

# sample rate
target_sr = 16000
# audio splitting duration (seconds)
time_units = 6

processor = Wav2Vec2Processor.from_pretrained(model_name, use_auth_token=api_token, cache_dir=cache_dir)
model = Wav2Vec2ForCTC.from_pretrained(model_name, use_auth_token=api_token, cache_dir=cache_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


def convert(inputfile, outfile):
    data, sample_rate = librosa.load(inputfile)
    data = librosa.resample(data, orig_sr=sample_rate, target_sr=target_sr)
    sf.write(outfile, data, target_sr)

def split_file(file_path):
    audio, samplerate = sf.read(file_path)
    segment_duration = time_units * samplerate
    
    count = 1
    for i in range(0, len(audio), segment_duration):
        segment = audio[i:i+segment_duration]
        sf.write(f"segment{count}.wav", segment, samplerate)
        count += 1
    
    return count - 1


def parse_transcription(wav_file_path):
    filename = wav_file_path.split('.')[0]
    new_file_path = filename + "16k.wav"

    convert(wav_file_path, new_file_path)
    audio_file_length = split_file(new_file_path)

    transcription = ""
    for i in range(1, audio_file_length + 1):
        # print(i)
        speech, _ = sf.read(f"segment{i}.wav")
        input_values = processor(speech, sampling_rate=16_000, return_tensors="pt").input_values
        input_values = input_values.to(device)
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription += processor.decode(predicted_ids[0], skip_special_tokens=True)

    return transcription

examples = [
    "sample_javanese_01.wav",
    "WIKITONGUES - Nila speaking Javanese.wav",
    "WIKITONGUES - Ayu speaking Javanese.wav",   
]

print(parse_transcription(examples[2]))