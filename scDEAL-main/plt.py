import matplotlib.pyplot as plt
import re

def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    epochs = []
    train_losses = []
    val_losses = []

    for line in lines:
        match_epoch = re.search(r'Epoch (\d+)/', line)
        match_train_loss = re.search(r'train Loss: (\d+\.\d+)', line)
        match_val_loss = re.search(r'val Loss: (\d+\.\d+)', line)

        if match_epoch:
            epochs.append(int(match_epoch.group(1)))
        if match_train_loss:
            train_losses.append(float(match_train_loss.group(1)))
        if match_val_loss:
            val_losses.append(float(match_val_loss.group(1)))

    return epochs, train_losses, val_losses

def plot_losses(epochs, train_losses, val_losses):
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()
def convert_backslashes_to_slashes(file_path):
    return file_path.replace('\\', '/')

# 文件路径替换为你的实际文件路径
file_path = 'E:\code\scDEAL-main\save\logs/2023-10-24-20-14-04transfer/2023-10-24-20-14-04transfer.log'
file_path = convert_backslashes_to_slashes(file_path)
epochs, train_losses, val_losses = parse_log_file(file_path)
plot_losses(epochs, train_losses, val_losses)