import csv
import os
import time
from datetime import datetime
from uuid import uuid4
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class VoiceDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        mfcc = torch.from_numpy(self.dataframe.iloc[idx]['mfcc']).float()
        onset = self.dataframe.iloc[idx]['onset']  # better float, could be int
        return mfcc, onset


"""
# CNN-LSTM
class ComplexVoiceDetectionModel(nn.Module):
    def __init__(self):
        super(ComplexVoiceDetectionModel, self).__init__()
        self.conv1 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.lstm1 = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 256, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
"""


# Pure CNN
class ModifiedVoiceDetectionModel(nn.Module):
    def __init__(self):
        super(ModifiedVoiceDetectionModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=44928, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


"""
# Regression CNN
class Regression1DCNN(nn.Module):
    def __init__(self):
        super(Regression1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=13, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(179968, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
"""

"""
# RNN(GRU)
class GRUVoiceDetectionModel(nn.Module):
    def __init__(self):
        super(GRUVoiceDetectionModel, self).__init__()
        self.gru1 = nn.GRU(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=32, hidden_size=128, num_layers=1, batch_first=True)
        self.gru3 = nn.GRU(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x, _ = self.gru3(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
"""


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# find dot wav file with file name in a given directory and return its absolute path
def find_wav_new(name, roots, files):
    if name in files:
        return os.path.join(roots[files.index(name)], name)


def pre_emphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


# load, preprocess and mfcc feature extraction of audio
def process_audio(wav_path, max_sequence_length):
    y, sr = librosa.load(wav_path, sr=None)  # load audio

    # Resampling if audio's sr does not match 48khz
    target_sr = 48000
    if sr != 48000:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # Padding
    if len(y) < max_sequence_length * sr:
        pad_width = max_sequence_length * sr - len(y)
        y = np.pad(y, pad_width=(0, pad_width))

    # print(y.shape)
    # pre-emphasis
    y_emp = pre_emphasis(y)

    # MFCC feature extraction
    # MFCC config
    n_mfcc = 64  # number of mfcc feature
    window_length = 512
    hop_length = int(window_length / 2)
    n_fft = int(window_length)
    n_mels = 64  # number of Mel filter
    fmax = sr * 0.5

    # perform mfcc
    mfcc = librosa.feature.mfcc(y=y_emp, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, n_mels=n_mels, fmax=fmax, hop_length=hop_length, window='hamming')
    # mfcc = mfcc.transpose()
    # print(mfcc.shape)
    return y, mfcc


def train(model, train_loader, criterion, optimizer, device, dataset_length):
    model.train()
    train_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.float().to(device), targets.float().to(device)
        # print(inputs.shape)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets.squeeze())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / dataset_length


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.float().to(device), targets.float().to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.squeeze())

            mae = torch.abs(outputs.squeeze() - targets.squeeze()).mean()
            mse = ((outputs.squeeze() - targets.squeeze()) ** 2).mean()

            total_loss += loss.item()
            total_mae += mae.item()
            total_mse += mse.item()
            total_samples += len(inputs)

    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    avg_mse = total_mse / total_samples

    return avg_loss, avg_mae, avg_mse


def analyze_model_performance(file_path):
    data = pd.read_csv(file_path)
    filtered_data = data[data['NO'] > data['NO'].max() * 0.5]

    validation_loss_mean = filtered_data['Validation Loss'].mean()
    validation_mae_mean = filtered_data['Validation MAE'].mean()

    suitable_models = filtered_data[(filtered_data['Validation Loss'] < validation_loss_mean) &
                                    (filtered_data['Validation MAE'] < validation_mae_mean)]

    best_models = suitable_models.sort_values(by=['Validation Loss', 'Validation MAE'])

    print("Top performing models:")
    print(best_models.head())


def main():
    root_dir = r"/Users/taiyuan/Desktop/onsetEEG/behavioral"  # directory where all dot wav files are stored
    model_dir = r"./model"  # directory for storing trained weights and log
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Index all files in root directory
    roots, files = [], []
    for root, _, file1 in os.walk(root_dir):
        for filesub in file1:
            files.append(filesub)
            roots.append(root)

    # Read VOT mark result CSV
    print('Reading CSV...')
    results_csv_path, wav_files, wav_onset = [], [], []
    root_ctn = os.listdir(root_dir)
    for ctn in root_ctn:
        if ctn.endswith('.csv'):
            results_csv_path.append(os.path.join(root_dir, ctn))

    # load dot wav file name and onset time to pandas dataFrame
    for csv_file in results_csv_path:
        with open(csv_file, 'r') as csv_in:
            lines = csv.reader(csv_in)
            next(lines)
            for line in lines:
                if line[3] == "1" and line[2] != '' and line[2] != '--undefined--' and line[1] != '':
                    wav_files.append(line[1])
                    wav_onset.append(float(line[2]))

    dataset = pd.DataFrame({'wav': wav_files, 'onset': wav_onset})

    mfcc_list = []  # list for storing MFCC features
    signal_list = []  # list for storing raw audio data time series
    max_sequence_length = 15  # max audio length in seconds

    # Porecess audio data
    print('Loading and preprocessing audio(s)...')
    for idx, wav in enumerate(dataset.wav):
        y, mfcc = process_audio(find_wav_new(wav, roots, files), max_sequence_length)
        mfcc_list.append(mfcc)
        signal_list.append(y)

    dataset['mfcc'] = mfcc_list
    dataset['signal'] = signal_list
    print(dataset)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # GPU accelaration with Apple M-series chipset
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU accelaration with Nvidia graphic cards

    # Prepare dataset
    train_data, test_data = train_test_split(dataset, test_size=0.1, random_state=42)  # split train:test = 9:1
    train_dataset = VoiceDataset(train_data)
    test_dataset = VoiceDataset(test_data)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model initialization
    model = ModifiedVoiceDetectionModel().to(device)
    
    # Continued training
    continued_train = 'n'
    if os.path.exists('./vot_best.pth'):
        continued_train = str(input('vot_best.pth found, do you wanna continue training(y/n)'))
        if continued_train == 'y':
            model.load_state_dict(torch.load('./vot_best.pth', map_location=device))
        else:
            pass
    
    criterion = nn.MSELoss()  # Use MSE loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.1
    early_stopping = EarlyStopping(patience=10, delta=0)  # Early stopping config

    # Training
    # create folder for output
    unique_id = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
    if not os.path.exists(os.path.join(model_dir, unique_id)):
        os.makedirs(os.path.join(model_dir, unique_id))
    output_dir = os.path.join(model_dir, unique_id)
    
    epoch = 0
    all_metrics = []  # store metrics for model performance evaluation
    val_loss_min = 0
    
    if continued_train == 'y':
        val_loss, val_mae, val_mse = evaluate(model, test_loader, criterion, device)
        val_loss_min = val_loss
    
    print('\nStart training...\n')
    while not early_stopping.early_stop:
        train_loader_tqdm = tqdm(train_loader, desc=f"[{epoch + 1}]", total=len(train_loader), colour='GREEN')
        train_loss = train(model, train_loader_tqdm, criterion, optimizer, device, len(train_loader.dataset))
        val_loss, val_mae, val_mse = evaluate(model, test_loader, criterion, device)  # evaluate model performance every epoch
        print(f'[{epoch + 1}]: Train loss: {train_loss} | Validation Loss: {val_loss} | Validation MAE: {val_mae} | Validation MSE: {val_mse}')

        all_metrics.append({
            "NO": len(all_metrics) + 1,
            "Epoch": f"{epoch + 1}",
            "Train Loss": train_loss,
            "Validation Loss": val_loss,
            "Validation MAE": val_mae,
            "Validation MSE": val_mse
        })
        torch.save(model.state_dict(), os.path.join(output_dir, f'model_epoch_{epoch}_{val_loss}.pth'))
        
        # Save the best model
        if epoch == 0 and continued_train == 'n':
            val_loss_min = val_loss
        else:
            if val_loss < val_loss_min:
                val_loss_min = val_loss
                torch.save(model.state_dict(), os.path.join(output_dir, f'vot_best.pth'))
                
        early_stopping(val_loss)  # Check early stopping conditions
        epoch += 1

    print(f"\nTraining complete.")

    # Save and analyze model metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(output_dir, 'training_metrics.csv'), index=False)
    analyze_model_performance(os.path.join(output_dir, 'training_metrics.csv'))


if __name__ == '__main__':
    main()
