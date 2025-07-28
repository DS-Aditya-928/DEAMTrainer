import numpy as np
import librosa
import soundfile
import os
import pandas
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchaudio as ta
import torchaudio.transforms as T
from torch import nn

class SongObject:
  samples = []#each is a 5sec clip converted to mel spectrogram
  sampleRate = 0
  arousal = 0.0
  valence = 0.0
  def __init__(self) -> None:
    self.samples = []
    self.sampleRate = 0
    arousal = 0.0
    valence = 0.0
    pass

print("DEAMTrainer")
  
ta.list_audio_backends()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

dataRoot = "X:\\Downloads\\archive\\"
songSD ="DEAM_audio\\MEMD_audio\\"
annoSD = "DEAM_Annotations\\annotations\\annotations averaged per song\\song_level\\static_annotations_averaged_songs_1_2000.csv"

songData = {}
df = pandas.read_csv(str(dataRoot + annoSD))

SAMPLE_RATE = 44100

#build transformation pipeline
transform = torch.nn.Sequential(
    T.MelSpectrogram(
        sample_rate=44100,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        power=2.0  # power=2.0 for power spectrogram (use with AmplitudeToDB)
    ),
    T.AmplitudeToDB(stype="power", top_db=80)  # convert to dB
)

transform = transform.to(device)

for i in tqdm(range(1, 11)):#last 59 are long
  mPath = str(dataRoot + songSD + str(i) + ".mp3")
  if(os.path.exists(mPath)):
    #find deets in csv
    row = df[df['song_id'] == i]
    samples, sampleRate = ta.load(uri = mPath, normalize = True, backend = 'soundfile')
    samples = samples.to(device)

    if(sampleRate != SAMPLE_RATE):
      continue

    if(samples.shape[0] > 1): #not mono
      samples = torch.mean(samples, 0, keepdim = True)

    toAdd = SongObject()
    toAdd.arousal = row[' arousal_mean'].values[0]
    toAdd.valence = row[' valence_mean'].values[0]

    SAMPLE_LEN = 5
    NUM_SAMPLES = SAMPLE_LEN * sampleRate

    toAdd.sampleRate = sampleRate

    for k in range(0, len(samples[0]), NUM_SAMPLES):
      s = samples[0][k:k+NUM_SAMPLES]

      if((len(s) >= NUM_SAMPLES)):
        toAdd.samples += [transform(s)]
      del(s)
    songData[i] = toAdd

#atleast it caches them
print("\nLoaded!")

print(songData.keys())

#samples, sr = librosa.load(mPath, sr=44100, mono=True)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 2)

displaySong = 10

print(str(songData[displaySong].sampleRate) + " Valence: " + str(songData[displaySong].valence)
                                      + " Arousal: " + str(songData[displaySong].arousal))

#librosa.display.specshow(songData[displaySong].samples[0], sr=songData[displaySong].sampleRate, hop_length=512, x_axis='time', y_axis='mel')
x = songData[displaySong].samples[0].cpu()
plt.imshow(x, aspect='auto', origin='lower', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title(f'Mel-Spectrogram for Song ID {displaySong} (First 5s Segment)')
plt.tight_layout()

plt.show()
print(songData[displaySong].samples[0].unsqueeze(0).shape)