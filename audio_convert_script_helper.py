import librosa
import soundfile as sf

def convert(inputfile, outfile):
    target_sr = 16000
    data, sample_rate = librosa.load(inputfile)
    data = librosa.resample(data, orig_sr=sample_rate, target_sr=target_sr)
    sf.write(outfile, data, target_sr)

examples = [
    "sample_javanese_01.wav",
    "WIKITONGUES - Nila speaking Javanese.wav",
    "WIKITONGUES - Ayu speaking Javanese.wav",
]

for file_path in examples:
    filename = file_path.split('.')[0]
    convert(file_path, filename + "_converted.wav")