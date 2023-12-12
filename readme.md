# MandVOT: Mandarin Voice Onset Time (VOT) Detection Using Machine Learning
MandVOT is a machine learning model, employing a Convolutional Neural Network (CNN) architecture, trained for the automated detection of Voice Onset Time (VOT) in Mandarin speech.
## Background
### Dataset
### Network Structure
```
INPUT (MFCC Features, np.array, [64, 2813])
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
```
### Workflow
#### Audio preprocessing and feature extraction
```
START
|
|--- Read VOT annotaion CSV(s)
|--- Load Audio (wav path from CSV(s))
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
END
```
## Performance
## Install
### Standard
### Praat Plugin
### Javascript
## Usage
## License
[MIT](./LICENSE) © Ryan Alloriadonis
