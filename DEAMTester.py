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
import random
from sklearn.metrics import r2_score

class MoodDEAM(nn.Module):
  def __init__(self):
    super().__init__()
    self.modelArch = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=3), 
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(64, 128, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(128, 256, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(256, 512, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(512, 1024, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),

    nn.Linear(1024 * 2 * 11, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 2),
    nn.Sigmoid()
    ).to(device)

  def forward(self, input):
    return self.modelArch(input)


device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
print(device)

model = MoodDEAM()
model.load_state_dict(torch.load("C:\\NNModels\\DEAM.pth", weights_only=True))
model.eval()

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
        power=2.0 
    ),
    T.AmplitudeToDB(stype="power", top_db=80)
)

transform = transform.to(device)

sampleStack = []
avStack = []

'''
for i in tqdm(range(1, 200)):#last 59 are long
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

    #toAdd = SongObject()
    arousal = ((row[' arousal_mean'].values[0])/10.0).astype(np.float32)
    valence = ((row[' valence_mean'].values[0])/10.0).astype(np.float32)

    SAMPLE_LEN = 5
    NUM_SAMPLES = SAMPLE_LEN * sampleRate

    #toAdd.sampleRate = sampleRate

    for k in range(0, len(samples[0]), NUM_SAMPLES):
      s = samples[0][k:k+NUM_SAMPLES]

      if((len(s) >= NUM_SAMPLES)):
        sampleStack += [transform(s).unsqueeze(0)]
        avStack += [(arousal, valence)]
      del(s)
    
    #songData[i] = toAdd
'''
mPath = "X:\\Downloads\\hartebeest.mp3"
samples, sampleRate = ta.load(uri = mPath, normalize = True, backend = 'soundfile')
resampler = ta.transforms.Resample(orig_freq=sampleRate, new_freq=44100)
samples = resampler(samples)
samples = samples.to(device)

print(sampleRate)

SAMPLE_LEN = 5
NUM_SAMPLES = SAMPLE_LEN * sampleRate

for k in range(0, len(samples[0]), NUM_SAMPLES):
      s = samples[0][k:k+NUM_SAMPLES]

      if((len(s) >= NUM_SAMPLES)):
        sampleStack += [transform(s).unsqueeze(0)]
        #avStack += [(arousal, valence)]
      del(s)

#atleast it caches them
print("\nLoaded!")
print(len(sampleStack))
#print(len(avStack))

testLabelsP = []
testLabelsC = []
predictions = []
for k in tqdm(range(len(sampleStack))):
  #print("Testing on ID: " + str(k))
    with torch.no_grad():
      predictions += [model(sampleStack[k].unsqueeze(0).to(device))] 
      '''
      outp = avStack[k]
      testLabelsP += [prediction.cpu().numpy().flatten()]
      testLabelsC += [(outp)]
      '''
    #print(str(prediction[0]) + " vs " + str(outp))

x = 0
for i in predictions:
    print("At " + str(x) + "s: " + str(i))
    x += 5

'''
testLabelsC = np.array(testLabelsC)
testLabelsP = np.array(testLabelsP)
  #print(testLabelsC)
  #print(testLabelsP)
r2_arousal = r2_score(testLabelsC[:, 0], testLabelsP[:, 0])
r2_valence = r2_score(testLabelsC[:, 1], testLabelsP[:, 1])

print(f"R2 (Arousal): {r2_arousal:.4f}")
print(f"R2 (Valence): {r2_valence:.4f}")
'''