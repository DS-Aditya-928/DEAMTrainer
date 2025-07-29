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
from sklearn.model_selection import train_test_split

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

for i in tqdm(range(1, 20)):#last 59 are long
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

    sampleStack = []
    for k in range(0, len(samples[0]), NUM_SAMPLES):
      s = samples[0][k:k+NUM_SAMPLES]

      if((len(s) >= NUM_SAMPLES)):
        sampleStack += [transform(s).unsqueeze(0)]
      toAdd.samples = torch.stack(sampleStack)  # stack all clips into a single tensor
      del(s)
    songData[i] = toAdd

#atleast it caches them
print("\nLoaded!")

print(songData.keys())
print(songData[10].samples.shape) 

#samples, sr = librosa.load(mPath, sr=44100, mono=True)

'''
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
#
'''

validIDs = list(songData.keys())
trainIDs, testIDs = train_test_split(validIDs, test_size=0.3)


class MoodDEAM(nn.Module):
  def __init__(self):
    super().__init__()
    self.modelArch = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3),  # Expects input of shape [B, 1, 128, 431]
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(64, 128, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),

    nn.Linear(128 * 14 * 52, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)
    ).to(device)

  def forward(self, input):
    return self.modelArch(input)


model = MoodDEAM().to(device)
#print(model)

lossFunc = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 10

for i in range(EPOCHS):
  print("Epoch: " + str(i + 1))

  for j in trainIDs:
    #all the clips should already be in gpu
    prediction = model(songData[j].samples)
    print(prediction.shape)
    print(prediction)