import os
import numpy as np
import pickle
import argparse
from utils import cross_entropy, save_args, plot_metrics, visualize_intermediate_activations
from tqdm import tqdm
from utils import load_cifar10_multi_batch, one_hot
from dataloader import Dataloader
from model import MyModel
from optimizer import my_SGD
from train import train, validate


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', default=0, type=int)
    parser.add_argument('--data_dir', default='./data/cifar-10-batches-py')
    parser.add_argument('--logpath', default='./results/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--in_channel', type=int, default=3072)
    parser.add_argument('--hiddenlayer_1', type=int, default=1024)
    parser.add_argument('--hiddenlayer_2', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--total_epochs', type=int, default=100)
    parser.add_argument('--epochs_decay', type=int, default=50)
    parser.add_argument('--decay', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=926)
    parser.add_argument('--date', default='0413')
    parser.add_argument('--plot', default=True)
    parser.add_argument('--activation', type=str, default='relu')
    args = parser.parse_args()
    return args


def main(args):
    args_name = f'./configs/Task_{args.task_id}_{args.date}_config.json'
    folder_path = os.path.dirname(args_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    save_args(args, args_name)
    np.random.seed(args.seed)
    log_dir = args.logpath + 'Task_{}_{}_{}_{}_{}_{}/'.format(args.task_id, args.learning_rate, args.activation, args.hiddenlayer_1, args.hiddenlayer_2, args.weight_decay)
    data_dir = args.data_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    batch_idx_list_train = [1,2,3,4]
    batch_idx_list_val = [5]
    
    model = MyModel(in_channel=args.in_channel, hidden_layer=[args.hiddenlayer_1, args.hiddenlayer_2], num_classes=args.num_classes, activation=args.activation)
    optimizer = my_SGD(parameters=model.parameters,learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    criterion = cross_entropy
    
    X_train, y_train = load_cifar10_multi_batch(data_dir, batch_idx_list_train)
    X_val, y_val = load_cifar10_multi_batch(data_dir, batch_idx_list_val)
    y_train = one_hot(y_train, args.num_classes)
    y_val = one_hot(y_val, args.num_classes)
    train_loader = Dataloader(X_train, y_train, batch_size=args.batch_size, shuffle=True)
    val_loader = Dataloader(X_val, y_val, batch_size=args.batch_size, shuffle=False)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_acc = 0.0
    for epoch in tqdm(range(args.total_epochs), desc='Epochs'):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoints = {'parameters':model.parameters,
                        'acc':val_acc,
                        'epoch': epoch}
            with open(log_dir + f'best_model.pkl', 'wb') as f:
                pickle.dump(checkpoints, f)
            print(f'New model is saved!')
            
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        if (epoch + 1) % args.epochs_decay == 0:
            optimizer.adjust_learning_rate()
            print('Learning rate decay: {}'.format(optimizer.learning_rate))
        if args.plot:
            plot_metrics(train_losses, val_losses, train_accs, val_accs, log_dir)
        if epoch == args.total_epochs - 1 and args.plot:
            visualize_intermediate_activations(model,
                                               X_val[:5],
                                               y_val[:5],
                                               save_dir=log_dir,
                                               epoch=epoch,
                                               class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'])


if __name__ == '__main__':
    args = getArgs()
    main(args)
