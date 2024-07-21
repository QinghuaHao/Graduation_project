import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from tqdm import tqdm
import wandb
#LSTM 模型准确率高，预测准确
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


class MotionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(MotionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.relu = nn.ReLU()  # Add ReLU activation function
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)  # Apply ReLU activation
        out = self.dropout(out)
        out = self.fc2(out)
        return out



def load_and_preprocess_generated_data(file_paths):
    data_frames = []
    labels = []
    for i, path in enumerate(file_paths):
        df = pd.read_csv(path)
        df['label'] = i
        data_frames.append(df)
    combined_df = pd.concat(data_frames)
    combined_df = combined_df[['x', 'y', 'label']]

    sequences = []
    sequence_length = 60
    num_sequences = 150

    for label in combined_df['label'].unique():
        group = combined_df[combined_df['label'] == label]
        for i in range(0, len(group), sequence_length * num_sequences):
            segment = group.iloc[i:i + sequence_length * num_sequences]
            if len(segment) == sequence_length * num_sequences:
                for j in range(num_sequences):
                    start = j * sequence_length
                    end = start + sequence_length
                    vectors = segment[['x', 'y']].values[start:end]
                    sequences.append((vectors, label))

    data = np.array([s[0] for s in sequences])
    labels = np.array([s[1] for s in sequences])
    return data, labels


def preprocess_input_data(input_path):
    df = pd.read_csv(input_path)
    if len(df) < 60:
        raise ValueError(f"Input data size {len(df)} is less than the required size 60")
    data = df[['x', 'y']].values[:60].reshape(1, 60, 2)
    return torch.tensor(data, dtype=torch.float32)


def train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=25):
    train_loss_history = []
    train_accuracy_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(
            train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
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


def evaluate_model(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    loss = running_loss / len(test_loader.dataset)
    wandb.log({'test_loss': loss, 'test_accuracy': accuracy})
    print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
    return accuracy, loss


def predict(model, input_data):
    input_data = input_data.to(device)
    with torch.no_grad():
        output = model(input_data)
        _, predicted = torch.max(output, 1)
    return predicted.item()


def visualize_trajectory(data, labels, model, input_data, prediction, title):
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

    plt.show()


def visualize_training_progress(train_loss, train_accuracy, title):
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

    plt.show()


def objective(trial):
    input_size = 2  # x, y 坐标
    hidden_size = trial.suggest_int('hidden_size', 32, 128)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    num_classes = 4  # 圆形, 方形, 三角形, L形

    model = MotionLSTM(input_size, hidden_size,
                       num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=trial.suggest_float('lr', 1e-4, 1e-2))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_loss, train_accuracy = train_model(
        model, train_loader, criterion, optimizer, scheduler, num_epochs=25)
    accuracy, _ = evaluate_model(model, test_loader, criterion)

    return accuracy


if __name__ == "__main__":
    wandb.init(
        project="motion-trajectory-classification-LSTM",
        entity="qinghua_master_project",
        config={
            "learning_rate": 0.001,
            "architecture": "LSTM",
            "dataset": "Generated Motion Data LSTM",
            "epochs": 25,
            "hidden_size": 64,
            "num_layers": 2
        }
    )
    file_paths = [
        '../data/standard/generated_circle.csv',
        '../data/standard/generated_square.csv',
        '../data/standard/generated_triangle.csv',
        '../data/standard/generated_l_shape.csv'
    ]

    data, labels = load_and_preprocess_generated_data(file_paths)
    dataset = MotionDataset(data, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = 2  # x, y 坐标
    hidden_size = wandb.config.hidden_size
    num_layers = wandb.config.num_layers
    num_classes = 4  # 圆, 方, 三角形, L形

    model = MotionLSTM(input_size, hidden_size,
                       num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=wandb.config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    train_loss, train_accuracy = train_model(
        model, train_loader, criterion, optimizer, scheduler, num_epochs=wandb.config.epochs)
    test_accuracy, test_loss = evaluate_model(model, test_loader, criterion)

    # 保存训练的模型和超参数
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_classes': num_classes
    }, '../motion_lstm_model.pth')

    # 加载训练的模型并对新输入轨迹进行预测
    checkpoint = torch.load('../motion_lstm_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    input_path = '../data/input_data/input.csv'
    input_data = preprocess_input_data(input_path)
    prediction = predict(model, input_data)

    visualize_trajectory(data, labels, model, input_data,
                         prediction, title='Without Hyperparameter Tuning')
    visualize_training_progress(
        train_loss, train_accuracy, title='Without Hyperparameter Tuning')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    print(f'Best trial: {study.best_trial.value}')
    print(f'Best params: {study.best_trial.params}')

    best_params = study.best_trial.params
    model = MotionLSTM(
        input_size,
        best_params['hidden_size'],
        best_params['num_layers'],
        num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])

    train_loss_tuned, train_accuracy_tuned = train_model(
        model, train_loader, criterion, optimizer, scheduler, num_epochs=25)
    test_accuracy_tuned, test_loss_tuned = evaluate_model(
        model, test_loader, criterion)

    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': best_params['hidden_size'],
        'num_layers': best_params['num_layers'],
        'num_classes': num_classes,
        'lr': best_params['lr']
    }, '../motion_lstm_model_tuned.pth')

    checkpoint = torch.load('../motion_lstm_model_tuned.pth')
    model = MotionLSTM(
        checkpoint['input_size'],
        checkpoint['hidden_size'],
        checkpoint['num_layers'],
        checkpoint['num_classes']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    prediction_tuned = predict(model, input_data)
    visualize_trajectory(data, labels, model, input_data,
                         prediction_tuned, title='With Hyperparameter Tuning')
    visualize_training_progress(
        train_loss_tuned,
        train_accuracy_tuned,
        title='With Hyperparameter Tuning')

    def use_model(model_path, input_path):
        checkpoint = torch.load(model_path)
        model = MotionLSTM(
            checkpoint['input_size'],
            checkpoint['hidden_size'],
            checkpoint['num_layers'],
            checkpoint['num_classes']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        input_data = preprocess_input_data(input_path)
        prediction = predict(model, input_data)

        labels_text = ['Circle', 'Square', 'Triangle', 'L Shape']
        print(f"输入轨迹是一个{labels_text[prediction]}。")

    use_model('../motion_lstm_model_tuned.pth', '../data/input_data/input.csv')
