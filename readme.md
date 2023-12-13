# MandVOT: Mandarin Voice Onset Time (VOT) Detection Using Machine Learning
MandVOT is a machine learning model, employing a Convolutional Neural Network (CNN) architecture, trained for the automated detection of Voice Onset Time (VOT) in Mandarin speech.
## Background
### Dataset Overview
  - **Source**
    - Institution: ``Department of Chinese and Bilingual Studies, Hong Kong Polytechnic University``
    - Research Lead: ``Dr. Xiaocong Chen`` [Github Profile](https://github.com/felcshallot) [Google Scholar](https://scholar.google.com/citations?user=gHlLwKoAAAAJ&hl=en)
  - **Dataset Description**
    - This collection comprises a total of 12,522 audio recordings in WAV format, sampled at 48kHz. These recordings were captured as part of an EEG study focusing on Mandarin speech.
  - **Speaker Details**
    - Number of Speakers: ``38``
    - Language: ``Mandarin``
  - **Annotations**
    - Each recording is accompanied by precise Voice Onset Time (VOT) annotations. These annotations have been meticulously marked using [Praat](https://www.fon.hum.uva.nl/praat/) by Dr.Xiaocong CHEN and others.
### Network Structure
```
INPUT <MFCC Features, np.array, [64, 2813]>
|
|-- -Conv1 (32 filters, kernel=3, ReLU)
|       |--- Max Pooling (kernel=2)
|
|--- Conv2 (64 filters, kernel=3, ReLU)
|       |--- Max Pooling (kernel=2)
|
|--- Fully Connected (128 units, ReLU)
|--- Fully Connected (1 unit)
|
OUTPUT (VOT prediction in ms, float)

Number of parameters: 5763425
```
### Workflow
#### Dataset Preparation
  - Reading
    ```
    START <dataset, pd.dataFrame, [0, 0]>
    |
    |--- Read VOT annotaion CSV(s) <dataset, pd.dataFrame, [2('wav','onset'), N_audio]>
    |--- Load Audio (wav path from CSV(s))
    |       |--- Read raw audio signal
    |       |--- Check Sample Rate (sr)
    |       |       |--- Resample to 48kHz if sr != 48000
    |       |
    |       |--- Padding (Zero-padding)
    |       |--- Apply Pre-emphasis (y_emp = y[0] + y[1:] - alpha * y[:-1])
    |       |--- Perform MFCC Feature Extraction
    |               |--- Configuration:
    |               |       - Number of MFCC features (n_mfcc): 64
    |               |       - Window length: 512
    |               |       - Hop length: window_length / 2
    |               |       - Number of FFT points (n_fft): window_length
    |               |       - Number of Mel filter banks (n_mels): 64
    |               |       - Maximum frequency (fmax): sr * 0.5
    |               |       - Window function: 'hamming'
    |               |
    |               |--- Compute MFCC Features (librosa.feature.mfcc)
    |
    |--- Return Processed Audio (y, np.array, [1, 720000]) and MFCC Features (mfcc, np.array, [64, 2813])
    |
    END <dataset, pd.dataFrame, [4('wav','onset','signal','mfcc'), N_audio]>
    ```
  - Cleaning
  - Augmentation
  - Train/test splitting
#### Dataset preparation

#### Model initializtion

#### Training

#### Evaluation

## Performance

## Install
### Standard

### Praat Plugin
In progress...

### Javascript
In progress...

## Usage

## License
[MIT](./LICENSE) © Ryan Alloriadonis
