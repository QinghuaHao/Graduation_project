import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import optuna
from sklearn.metrics import precision_recall_fscore_support
# Use Metal backend on MacBook M1
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class MotionDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label

class LSTMCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMCNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size * 2, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(128 * (60 // 4), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out.permute(0, 2, 1)  # Change shape to (batch, hidden_size*2, sequence_length) for CNN
        out = self.cnn(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out

def load_and_preprocess_generated_data(file_paths, sequence_length=60, num_sequences=900):
    data_frames = []
    for i, path in enumerate(file_paths):
        df = pd.read_csv(path)
        df['label'] = i
        data_frames.append(df)

    combined_df = pd.concat(data_frames)
    combined_df = combined_df[['x', 'y', 'label']]

    sequences = []
    for label in combined_df['label'].unique():
        group = combined_df[combined_df['label'] == label]
        real_data = group.iloc[:sequence_length * num_sequences]
        augmented_data = group.iloc[sequence_length * num_sequences: 2 * sequence_length * num_sequences]

        # real data
        for i in range(0, len(real_data), sequence_length):
            segment = real_data.iloc[i:i + sequence_length]
            if len(segment) == sequence_length:
                vectors = segment[['x', 'y']].values
                sequences.append((vectors, label))

        # augmented data
        for i in range(0, len(augmented_data), sequence_length):
            segment = augmented_data.iloc[i:i + sequence_length]
            if len(segment) == sequence_length:
                vectors = segment[['x', 'y']].values
                sequences.append((vectors, label))

    data = np.array([s[0] for s in sequences])
    labels = np.array([s[1] for s in sequences])

    # Ensure no data leakage by shuffling before splitting
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data, labels = data[indices], labels[indices]

    # Split into training, validation, and test sets
    train_val_split = int(0.8 * len(data))
    val_test_split = int(0.1 * len(data)) + train_val_split
    train_data, train_labels = data[:train_val_split], labels[:train_val_split]
    val_data, val_labels = data[train_val_split:val_test_split], labels[train_val_split:val_test_split]
    test_data, test_labels = data[val_test_split:], labels[val_test_split:]

    return train_data, train_labels, val_data, val_labels, test_data, test_labels

def preprocess_input_data(input_path):
    df = pd.read_csv(input_path)
    if len(df) < 60:
        raise ValueError(f"Input data size {len(df)} is less than the required size 60")
    data = df[['x', 'y']].values[:60].reshape(1, 60, 2)
    return torch.tensor(data, dtype=torch.float32)

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=30):
    train_loss_history = []
    train_accuracy_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        train_loss_history.append(epoch_loss)
        train_accuracy_history.append(epoch_accuracy)
        wandb.log({'train_loss': epoch_loss, 'train_accuracy': epoch_accuracy, 'epoch': epoch})
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

        scheduler.step()

    return train_loss_history, train_accuracy_history


def evaluate_model(model, test_loader, criterion, prefix=''):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total
    loss = running_loss / len(test_loader.dataset)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None)
    wandb.log({f'{prefix}test_loss': loss, f'{prefix}test_accuracy': accuracy})
    print(f'{prefix} Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
    print(f'{prefix} Precision: {precision}')
    print(f'{prefix} Recall: {recall}')
    print(f'{prefix} F1 Score: {f1}')

    # Confusion Matrix
    labels_text = ['Circle', 'Square', 'Triangle', 'L Shape']
    unique_classes = np.unique(np.concatenate((all_labels, all_predictions)))
    if len(unique_classes) < len(labels_text):
        all_labels = np.append(all_labels, [label for label in range(len(labels_text)) if label not in unique_classes])
        all_predictions = np.append(all_predictions,
                                    [label for label in range(len(labels_text)) if label not in unique_classes])

    cm = confusion_matrix(all_labels, all_predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_text)
    display.plot()
    plt.title(f'{prefix} Confusion Matrix')
    plt.savefig(f'../data/out_put_image/LSTM_CNN_25/{prefix}confusion_matrix.png')
    plt.show()

    return accuracy, loss, precision, recall, f1


def visualize_metrics(precision, recall, f1, title, prefix=''):
    labels_text = ['Circle', 'Square', 'Triangle', 'L Shape']
    x = np.arange(len(labels_text))  # Label locations
    width = 0.25  # Bar width

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1, width, label='F1 Score')

    ax.set_xlabel('Classes')
    ax.set_ylabel('Scores')
    ax.set_title(f'{title} - Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_text)
    ax.legend()

    for i in range(len(precision)):
        ax.text(i - width, precision[i] + 0.01, f'{precision[i]:.2f}', ha='center', va='bottom')
        ax.text(i, recall[i] + 0.01, f'{recall[i]:.2f}', ha='center', va='bottom')
        ax.text(i + width, f1[i] + 0.01, f'{f1[i]:.2f}', ha='center', va='bottom')

    plt.savefig(f'../data/out_put_image/LSTM_CNN_25/{prefix}{title}_metrics.png')
    plt.show()

def predict(model, input_data):
    input_data = input_data.to(device)
    with torch.no_grad():
        output = model(input_data)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def visualize_trajectory(data, labels, model, input_data, prediction, title, prefix=''):
    # Visualize training data
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['r', 'b', 'g', 'y']
    markers = ['o', 's', '^', 'v']
    labels_text = ['Circle', 'Square', 'Triangle', 'L Shape']

    for i in range(4):
        ax[0].scatter(data[labels == i][:, :, 0].flatten(), data[labels == i][:, :, 1].flatten(),
                      c=colors[i], label=labels_text[i], marker=markers[i])

    ax[0].set_title('Training Data')
    ax[0].legend()

    # Visualize input trajectory
    input_data_np = input_data.squeeze().cpu().numpy().reshape(-1, 2)
    ax[1].plot(input_data_np[:, 0], input_data_np[:, 1],
               'g-', label='Input Trajectory')
    ax[1].set_title('Input Trajectory')
    ax[1].legend()

    # Prediction result
    predicted_label = labels_text[prediction]
    ax[2].plot(input_data_np[:, 0], input_data_np[:, 1],
               'g-', label=f'Predicted: {predicted_label}')
    ax[2].set_title(f'Prediction - {title}')
    ax[2].legend()
    plt.savefig(f'../data/out_put_image/LSTM_CNN_25/{prefix}{title}_training_progress.png')
    plt.show()

def visualize_training_progress(train_loss, train_accuracy, title, prefix=''):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(train_loss, label='Train Loss')
    ax[0].set_title(f'{title} - Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(train_accuracy, label='Train Accuracy')
    ax[1].set_title(f'{title} - Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    plt.savefig(f'../data/out_put_image/LSTM_CNN_25/{prefix}{title}_trajectory.png')
    plt.show()

def objective(trial):
    input_size = 2  # x, y coordinates
    hidden_size = trial.suggest_int('hidden_size', 64, 256)
    num_layers = trial.suggest_int('num_layers', 2, 5)
    num_classes = 4  # Circle, Square, Triangle, L Shape

    model = LSTMCNN(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=trial.suggest_float('lr', 1e-4, 1e-2))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=30)
    accuracy, _,_,_,_= evaluate_model(model, test_loader, criterion)

    return accuracy

if __name__ == "__main__":
    wandb.init(
        project="LSTM_CNN",
        entity="qinghua_master_project",
        config={
            "learning_rate": 0.001,
            "architecture": "LSTM-CNN",
            "dataset": "Generated Motion Data LSTM-CNN",
            "epochs": 30,
            "hidden_size": 128,  # change this to improve our model
            "num_layers": 2  # change this to improve our model
        }
    )

    file_paths = [
        '../data/standard_gestures/900_Data/circle_augmented_data.csv',
        '../data/standard_gestures/900_Data/square_augmented_data.csv',
        '../data/standard_gestures/900_Data/triangl_augmented_data.csv',
        '../data/standard_gestures/900_Data/L_shape_augmented_data.csv'
    ]

    sequence_length = 60

    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_and_preprocess_generated_data(file_paths)
    train_dataset = MotionDataset(train_data, train_labels)
    val_dataset = MotionDataset(val_data, val_labels)
    test_dataset = MotionDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = 2  # x, y coordinates
    hidden_size = wandb.config.hidden_size
    num_layers = wandb.config.num_layers
    num_classes = 4  # Circle, Square, Triangle, L Shape

    model = LSTMCNN(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=wandb.config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Train and evaluate model before hyperparameter tuning
    train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=wandb.config.epochs)
    test_accuracy, test_loss, precision_before, recall_before, f1_before = evaluate_model(model, test_loader, criterion,
                                                                                          prefix='before_')
    visualize_metrics(precision_before, recall_before, f1_before, title='Without Hyperparameter Tuning',
                      prefix='before_')

    # Save the trained model before hyperparameter tuning
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_classes': num_classes
    }, 'Saved_Model/motion_lstm_cnn_model_25.pth')

    # Load and predict using the saved model before hyperparameter tuning
    checkpoint = torch.load('Saved_Model/motion_lstm_cnn_model_25.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    input_path = '../data/standard_gestures/input/circle.csv'
    input_data = preprocess_input_data(input_path)
    prediction = predict(model, input_data)

    visualize_trajectory(train_data, train_labels, model, input_data, prediction, title='Without Hyperparameter Tuning', prefix='before_')
    visualize_training_progress(train_loss, train_accuracy, title='Without Hyperparameter Tuning', prefix='before_')

    # Hyperparameter tuning with Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    print(f'Best trial: {study.best_trial.value}')
    print(f'Best params: {study.best_trial.params}')

    best_params = study.best_trial.params
    model = LSTMCNN(input_size, best_params['hidden_size'], best_params['num_layers'], num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'])

    # Train and evaluate the model after hyperparameter tuning
    train_loss_tuned, train_accuracy_tuned = train_model(model, train_loader, criterion, optimizer, scheduler,
                                                         num_epochs=30)
    test_accuracy_tuned, test_loss_tuned, precision_after, recall_after, f1_after = evaluate_model(model, test_loader,
                                                                                                   criterion,
                                                                                                   prefix='after_')
    visualize_metrics(precision_after, recall_after, f1_after, title='With Hyperparameter Tuning', prefix='after_')
    # Save the trained model and hyperparameters after tuning
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': best_params['hidden_size'],
        'num_layers': best_params['num_layers'],
        'num_classes': num_classes,
        'lr': best_params['lr']
    }, 'Saved_Model/motion_lstm_cnn_model_tuned_25.pth')

    # Load and predict using the tuned model
    checkpoint = torch.load('Saved_Model/motion_lstm_cnn_model_tuned_25.pth')
    model = LSTMCNN(checkpoint['input_size'], checkpoint['hidden_size'], checkpoint['num_layers'], checkpoint['num_classes']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    prediction_tuned = predict(model, input_data)
    visualize_trajectory(train_data, train_labels, model, input_data, prediction_tuned, title='With Hyperparameter Tuning', prefix='after_')
    visualize_training_progress(train_loss_tuned, train_accuracy_tuned, title='With Hyperparameter Tuning', prefix='after_')

def use_model(model_path, input_path):
    checkpoint = torch.load(model_path)
    model = LSTMCNN(checkpoint['input_size'], checkpoint['hidden_size'], checkpoint['num_layers'], checkpoint['num_classes']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    input_data = preprocess_input_data(input_path)
    prediction = predict(model, input_data)

    labels_text = ['Circle', 'Square', 'Triangle', 'L Shape']
    print(f"The trajectory is one {labels_text[prediction]}.")

# Example usage:
# use_model('../motion_model_tuned_30.pth', '../data/standard_gestures/input/circle.csv')
