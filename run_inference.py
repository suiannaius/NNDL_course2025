from utils import *
from utils import load_cifar10_test_data
from dataloader import Dataloader
from model import MyModel
from train import inference


def main(task_id):
    args = load_args('./configs/' + f'Task_{task_id}_0413' + '_config.json')
    model = MyModel(in_channel=args.in_channel, hidden_layer=[args.hiddenlayer_1, args.hiddenlayer_2], num_classes=args.num_classes)
    log_dir = args.logpath + 'Inference_Task_{}_{}_{}_{}_{}_{}/'.format(args.task_id, args.learning_rate, args.activation, args.hiddenlayer_1, args.hiddenlayer_2, args.weight_decay)
    load_dir = args.logpath + 'Task_{}_{}_{}_{}_{}_{}/'.format(args.task_id, args.learning_rate, args.activation, args.hiddenlayer_1, args.hiddenlayer_2, args.weight_decay)
    
    X_test, y_test = load_cifar10_test_data(args.data_dir)
    y_test = one_hot(y_test, args.num_classes)
    test_loader = Dataloader(X_test, y_test, batch_size=args.batch_size, shuffle=False)
    test_acc = inference(model, test_loader, load_dir=load_dir, task_id=task_id)
    visualize_intermediate_activations(model,
                                        X_test[:5],
                                        y_test[:5],
                                        save_dir=log_dir,
                                        epoch=1,
                                        class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'])
    visualize_weights(model, save_dir=log_dir, epoch=None)
    return 0
    
if __name__ == '__main__':
    task_id = 9  # The best setting after hyper-parameter searching.
    main(task_id)
