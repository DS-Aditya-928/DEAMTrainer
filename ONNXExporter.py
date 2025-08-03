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
from sklearn.metrics import r2_score
import onnxruntime
import time as t

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
print(device)

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

model = MoodDEAM()
model.load_state_dict(torch.load("C:\\NNModels\\DEAM.pth", weights_only=True))
model.eval()

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
        sampleStack += [transform(s).unsqueeze(0)]#add mono channel
        #avStack += [(arousal, valence)]
      del(s)

print("\nLoaded!")
print(len(sampleStack))
sampleStack = torch.randn(4, 1, 128, 431).to(device)

'''
onnx_program = torch.onnx.export(model, args = sampleStack, dynamo=True, input_names=["input"],
    output_names=["output"],
    dynamic_shapes={
        "input": {0: "batch_size"},
    })
'''

#onnx_program.save("C:\\NNModels\\ONNX_DEAM.onnx")

ort_session = onnxruntime.InferenceSession(
    "C:\\NNModels\\ONNX_DEAM.onnx", providers=["CPUExecutionProvider"]
)

inputs = ort_session.get_inputs()

for i, inp in enumerate(inputs):
    print(f"Input {i}:")
    print(f"  Name : {inp.name}")
    print(f"  Shape: {inp.shape}")
    print(f"  Type : {inp.type}")

start = t.time()
onnxruntime_outputs = ort_session.run(None, {'input':np.array(sampleStack)})
end = t.time()
print(end - start)
print(onnxruntime_outputs)