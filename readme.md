# MandVOT: Mandarin Voice Onset Time (VOT) Detection Using Machine Learning
MandVOT is a machine learning model, employing a Convolutional Neural Network (CNN) architecture, trained for the automated detection of Voice Onset Time (VOT) in Mandarin speech.
## Background
### Dataset
### Network Structure
```
Input (MFCC Features)
|
|--> Conv1 (32 filters, kernel=3, ReLU)
|       |
|       |--> Max Pooling (kernel=2)
|
|--> Conv2 (64 filters, kernel=3, ReLU)
|       |
|       |--> Max Pooling (kernel=2)
|
|--> Fully Connected (128 units, ReLU)
|
|--> Fully Connected (1 unit)
|
Output (Voice Onset Prediction)
```
## Install
### Standard
### Praat Plugin
### Javascript
## Usage
## License
[MIT](../LICENSE) © Ryan Alloriadonis
