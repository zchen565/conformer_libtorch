import torch
import torchaudio
dataset_path = "./data"

# download = True to get file
ds = torchaudio.datasets.LIBRISPEECH(root = dataset_path, url = 'train-clean-100', folder_in_archive = 'LibriSpeech', download = False)

# 
from torch.utils.data import Subset
subset = Subset(ds, range(1000))

import torchaudio.transforms as T
from torchaudio.transforms import TimeMasking, FrequencyMasking

# the parameters are from the google paper
def preprocess(waveform, target_length = 255999): # 260000 will be 1601 # padding to [1,80,1600] for conformer input 
    current_length = waveform.shape[1]
    if current_length > target_length:
        waveform = waveform[:,:target_length]
    elif current_length < target_length:

        padding_size = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, padding_size))
    # Step 1: Extract 80-channel filterbank features in Conformer Paper
    n_mels = 80
    n_fft = 400 #  25ms window
    hop_length = 160 # 10ms stride
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    waveform = mel_spectrogram(waveform)

    # Step 2: Spec Aug in Conformer Paper
    freq_masking = FrequencyMasking(freq_mask_param=27)
    waveform = freq_masking(waveform)

    max_time_mask_size = int(0.05 * waveform.shape[1])  # pS * 语音长度

    for _ in range(10):
        time_masking = TimeMasking(time_mask_param=max_time_mask_size)
        waveform = time_masking(waveform)

    return waveform

print(preprocess(ds[0][0]).shape)
print(preprocess(ds[22][0]).shape)

# prepare dictionary
transcriptions = []
for i in range(len(ds)):
    transcriptions.append(ds[i][2])

word_list = []
for transcription in transcriptions:
    words = transcription.split()  
    word_list.extend(words)

word_set = set(word_list)
word_dict = {word: index for index, word in enumerate(word_set)}

word_dict['<PAD>'] = len(word_dict)
word_dict['<UNK>'] = len(word_dict)

import json
with open('pt/word_dict.json', 'w') as json_file:
    json.dump(word_dict, json_file)


# get text tensor
for i in range(1000):
    text_tensor = torch.tensor([], dtype=torch.int64)

    tt = subset[i][2].split()
    new_text = [word_dict[word] for word in tt]

    text_tensor = torch.cat((text_tensor, torch.tensor(new_text, dtype=torch.int64)))
    torch.save(text_tensor, f'pt/text/{i}.pt')