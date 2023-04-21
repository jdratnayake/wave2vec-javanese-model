import soundfile as sf
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import gradio as gr
import os


api_token = "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"
model_name = "indonesian-nlp/wav2vec2-indonesian-javanese-sundanese"
cache_dir = "/usr/src/app/model_cache"

processor = Wav2Vec2Processor.from_pretrained(model_name, use_auth_token=api_token, cache_dir=cache_dir)
model = Wav2Vec2ForCTC.from_pretrained(model_name, use_auth_token=api_token, cache_dir=cache_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


def convert(inputfile, outfile):
    target_sr = 16000
    data, sample_rate = librosa.load(inputfile)
    data = librosa.resample(data, orig_sr=sample_rate, target_sr=target_sr)
    sf.write(outfile, data, target_sr)


def parse_transcription(wav_file):
    filename = wav_file.split('.')[0]
    convert(wav_file, filename + "16k.wav")
    speech, _ = sf.read(filename + "16k.wav")
    input_values = processor(speech, sampling_rate=16_000, return_tensors="pt").input_values
    input_values = input_values.to(device)
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    return transcription

examples = [
    "sample_javanese_01.wav",
    "WIKITONGUES - Nila speaking Javanese.wav",
    "WIKITONGUES Test.wav"
]

print(parse_transcription(examples[0]))