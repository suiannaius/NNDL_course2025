import random
import itertools
from run_training import main, getArgs

param_grid = {
    'learning_rate': [1e-2, 1e-3, 1e-4],
    'hiddenlayer_1': [1024, 512, 256, 128, 64, 32],
    'hiddenlayer_2': [512, 256, 128, 64, 32],
    'activation': ['relu', 'sigmoid', 'tanh'],
    'weight_decay': [0, 1e-2, 1e-3, 1e-4]
}


def get_random_combinations(param_grid, num_combinations=10):
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return random.sample(param_combinations, num_combinations)


def search_hyperparameters():
    random_combinations = get_random_combinations(param_grid, num_combinations=10)
    
    for i, params in enumerate(random_combinations):
        print(f'\n====== Running combination {i+1}/10: {params} ======')
        args = getArgs()
        args.learning_rate = params['learning_rate']
        args.hiddenlayer_1 = params['hiddenlayer_1']
        args.hiddenlayer_2 = params['hiddenlayer_2']
        args.activation = params['activation']
        args.weight_decay = params['weight_decay']

        args.task_id = i + 100
        args.date = '0413_search'
        args.plot = False
        
        args.total_epochs = 1

        main(args)

if __name__ == '__main__':
    search_hyperparameters()
