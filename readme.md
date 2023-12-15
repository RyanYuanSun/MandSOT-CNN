[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/license-mit.svg)](https://forthebadge.com)

# MandSOT: Mandarin Speech Onset Time (VOT) Detection Using Machine Learning
MandSOT is a machine learning model, employing a Convolutional Neural Network (CNN) architecture, trained for the automated detection of Speech Onset Time (SOT) in Mandarin speech.
## Background
### Datasets
#### Mandarin Speeches
  - EEG Picture Naming Records
    - **Source**
      - Institution: ``Department of Chinese and Bilingual Studies, Hong Kong Polytechnic University``
      - Research Lead: ``Dr. Xiaocong Chen`` [Github Profile](https://github.com/felcshallot) [Google Scholar](https://scholar.google.com/citations?user=gHlLwKoAAAAJ&hl=en)
    - **Dataset Description**
      - This collection comprises a total of 12,522 audio recordings in WAV format, sampled at 48kHz. These recordings were captured as part of an EEG study focusing on Mandarin speech.
    - **Speaker Details**
      - Number of Speakers: ``38``
      - Language: ``Mandarin``
    - **Annotations**
      - Each recording is accompanied by precise Speech Onset Time (SOT) annotations. These annotations have been meticulously marked using [Praat](https://www.fon.hum.uva.nl/praat/) by Dr.Xiaocong CHEN and others.
#### Acoustic Noises
  - DEMAND Dataset
    - See [DEMAND Dataset](https://www.kaggle.com/datasets/chrisfilo/demand).
  - Other Noises
    - Background noise recorded in room ZB217 at UBSN, Hong Kong Polytechnic University with AC fan set to lvl 1, 2 and 3.
    - Background noise recorded in office GH709, Hong Kong Polytechnic University.
### Network Structure
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv1d-1             [-1, 32, 4096]          21,536
         MaxPool1d-2             [-1, 32, 2048]               0
            Conv1d-3             [-1, 64, 2046]           6,208
         MaxPool1d-4             [-1, 64, 1023]               0
            Conv1d-5             [-1, 32, 1021]           6,176
         MaxPool1d-6              [-1, 32, 510]               0
            Conv1d-7              [-1, 64, 508]           6,208
         MaxPool1d-8              [-1, 64, 254]               0
            Linear-9                  [-1, 128]       2,080,896
           Linear-10                    [-1, 1]             129
================================================================
Total params: 2,121,153
Trainable params: 2,121,153
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 3.50
Forward/backward pass size (MB): 3.75
Params size (MB): 8.09
Estimated Total Size (MB): 15.34
----------------------------------------------------------------
```
### Workflow
#### Dataset Preparation
    ```
    START <dataset, pd.dataFrame, [0, 0]>
    |
    |
    |--- Read SOT annotaion CSV(s) <dataset, pd.dataFrame, [2('wav','onset'), N_audio]>
    |--- Load Audio (wav path from CSV(s))
    |       |--- Read raw audio signal
    |       |--- Check Sample Rate (sr)
    |       |       |--- Resample to 48kHz if sr != 48000
    |       |
    |       |--- Data Augmentation (adding noise)
    |       |--- Padding (Zero-padding)
    |       |--- Apply Pre-emphasis (y_emp = y[0] + y[1:] - alpha * y[:-1])
    |       |--- Perform MFCC Feature Extraction
    |               |--- Configuration:
    |               |       - Number of MFCC features (n_mfcc): 32/64/128
    |               |       - Window length: 256/512/1024
    |               |       - Hop length: window_length / 2
    |               |       - Number of FFT points (n_fft): window_length
    |               |       - Number of Mel filter banks (n_mels): 32/64/128
    |               |       - Maximum frequency (fmax): 10000
    |               |       - Window function: 'hamming'
    |               |
    |               |--- Compute and Combine MFCC Features (librosa.feature.mfcc)
    |
    |--- Return MFCC Features (mfcc, np.array, [224, 4096])
    |
    |
    END <dataset, pd.dataFrame, [3('wav','onset','mfcc'), N]>
    ```
#### Model initializtion

#### Training

#### Evaluation

## Performance

## Install
### Python
```shell
pip install mandsot
```
### Praat Plugin
In progress...
### Javascript
```html
<script src="https://www.sunrays.top/js/mandsot.js"></script>
```
## Usage
### Train
#### Python
  - Prepare dataset
    - ```example.csv```
      ```
      wav_name                       onset  on/off
      ---------------------------------------------
      example_audio_1.wav            898    1
      example_audio_2.wav            1145   1
      example_audio_3.wav            764    1
      ...
      ---------------------------------------------
      ```

### Inference

## License
[MIT](./LICENSE) © Ryan Alloriadonis
