import numpy as np
import pickle
import os
import json
import argparse
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def softmax(x):
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    
def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-8))


def one_hot(labels, num_classes):
    return np.eye(num_classes)[labels] 


def batch_norm(x, epsilon=1e-8):
    mean = np.mean(x, axis=0)
    variance = np.var(x, axis=0)
    x_norm = (x - mean) / np.sqrt(variance + epsilon)
    return x_norm


def plot_metrics(train_loss, val_loss, train_acc, val_acc, log_dir):
    train_epochs = range(1, len(train_loss) + 1)
    val_epochs = range(1, len(val_loss) + 1)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_epochs, train_loss, label='Training Loss')
    plt.plot(val_epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=18)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.subplot(1, 2, 2)
    plt.plot(train_epochs, train_acc, label='Training Accuracy')
    plt.plot(val_epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy', fontsize=18)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    # plt.show()
    plt.savefig(log_dir + 'plot.png',dpi=300)
    plt.close()
    
    
def save_args(args, filepath='config.json'):
    with open(filepath, 'w') as f:
        json.dump(vars(args), f, indent=4)
        

def load_args(filepath='config.json'):
    with open(filepath, 'r') as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    for key, value in config.items():
        parser.add_argument(f'--{key}', default=value, type=type(value))
    args = parser.parse_args()
    return args
        
        
def load_cifar10_single_batch(file_path):
    """加载单个batch的CIFAR10数据"""
    with open(file_path, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
        data = dict[b'data']  # shape: (10000, 3072)
        labels = dict[b'labels']  # list of 10000
        data = data.astype(np.float32) / 255.0  # 归一化
        return data, np.array(labels)


def load_cifar10_multi_batch(data_dir, batch_idx_list):
    """加载CIFAR10多个batch的训练数据和测试数据, 用于训练及验证"""
    train_data = []
    train_labels = []
    for i in batch_idx_list:
        file_path = os.path.join(data_dir, f'data_batch_{i}')
        data, labels = load_cifar10_single_batch(file_path)
        train_data.append(data)
        train_labels.append(labels)
    X_train = np.concatenate(train_data)
    y_train = np.concatenate(train_labels)
    return X_train, y_train


def load_cifar10_test_data(data_dir):
    X_test, y_test = load_cifar10_single_batch(os.path.join(data_dir, 'test_batch'))
    return X_test, y_test


def visualize_intermediate_activations(model, X_sample, y_sample, save_dir, epoch, class_names=None):
    """
    可视化隐藏层输出为热图，每个样本一张图
    """
    _, cache = model.forward(X_sample)
    _, z1, a1, z2, a2, _, _ = cache

    os.makedirs(save_dir, exist_ok=True)

    for idx in range(min(5, X_sample.shape[0])):  # 每次展示前5张样本
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        # 输入图像还原显示
        img = X_sample[idx].reshape(3, 32, 32).transpose(1, 2, 0)  # 还原为 RGB 格式
        if np.max(img) <= 1.0:
            img = img * 255.0
        if np.min(img) < 0.0:
            img = (img + 1.0) * 127.5
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        axs[0].imshow(img)  # 若是 float，需要 *255 后 clip
        label = y_sample[idx].argmax() if y_sample.ndim == 2 else y_sample[idx]
        title = f"Label: {class_names[label]}" if class_names is not None else f"Label: {label}"
        axs[0].set_title(title)
        axs[0].axis('off')

        # 第一层激活值热图
        axs[1].imshow(a1[idx][None, :], cmap='viridis', aspect='auto')
        axs[1].set_title('Hidden Layer 1 Activation')
        axs[1].set_xlabel('Neuron Index')

        # 第二层激活值热图
        axs[2].imshow(a2[idx][None, :], cmap='viridis', aspect='auto')
        axs[2].set_title('Hidden Layer 2 Activation')
        axs[2].set_xlabel('Neuron Index')

        plt.suptitle(f"Sample {idx} - Epoch {epoch}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"activation_sample{idx}_epoch{epoch}.png"))
        plt.close()
        

def visualize_weights(model, save_dir, epoch=None):
    os.makedirs(save_dir, exist_ok=True)
    
    W1 = model.parameters['W1']  # shape: [3072, H1]
    H1 = W1.shape[1]
    
    # 可视化 W1：将每一列 reshape 为 32×32×3 显示
    n_cols = 8
    n_rows = (H1 + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    for i in range(H1):
        weight_img = W1[:, i].reshape(3, 32, 32).transpose(1, 2, 0)
        weight_img = (weight_img - weight_img.min()) / (weight_img.max() - weight_img.min() + 1e-8)  # normalize to [0,1]
        
        row, col = divmod(i, n_cols)
        axs[row, col].imshow(weight_img)
        axs[row, col].axis('off')

    for i in range(H1, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axs[row, col].axis('off')
    
    title = f"W1 Visualization at Epoch {epoch}" if epoch is not None else "W1 Visualization"
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"W1_visual_epoch{epoch if epoch else 'final'}.png"))
    plt.close()

    # W2, W3 用热图可视化
    for l, W in enumerate([model.parameters['W2'], model.parameters['W3']], start=2):
        plt.figure(figsize=(10, 5))
        plt.imshow(W, aspect='auto', cmap='bwr')
        plt.colorbar()
        plt.title(f"W{l} Weights Heatmap")
        plt.xlabel(f'Output neurons (layer {l})')
        plt.ylabel(f'Input neurons (layer {l-1})')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"W{l}_heatmap_epoch{epoch if epoch else 'final'}.png"))
        plt.close()

    