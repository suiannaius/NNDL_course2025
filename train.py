import pickle
import numpy as np
from utils import cross_entropy, batch_norm

def train(model,
          dataloader, 
          optimizer, 
          criterion):
    
    total = 0
    correct = 0
    running_loss = 0.
    
    for X_batch, y_batch in dataloader:
        total += X_batch.shape[0]
        X_batch = batch_norm(X_batch)
        output, cache = model(X_batch)
        loss = criterion(y_batch, output)
        grads = model.backward(y_batch, cache)
        optimizer.step(grads)
        prediction = np.argmax(output, axis=1)
        label = np.argmax(y_batch, axis=1)
        correct += np.sum(prediction == label)
        running_loss += loss.item()
    acc = 100. * correct / total
    print("Training: Loss:{:.3g}, Acc:{:.3g}%".format(running_loss / len(dataloader), acc))
    return running_loss / len(dataloader), acc


def validate(model,
             dataloader, 
             criterion):
    
    total = 0
    correct = 0
    running_loss = 0.
    
    for X_batch, y_batch in dataloader:
        total += X_batch.shape[0]
        X_batch = batch_norm(X_batch)
        output, _ = model(X_batch)
        loss = criterion(y_batch, output)
        prediction = np.argmax(output, axis=1)
        label = np.argmax(y_batch, axis=1)
        correct += np.sum(prediction == label)
        running_loss += loss
        
    acc = 100. * correct / total
    print("Validation: Loss:{:.3g}, Acc:{:.3g}%".format(running_loss / len(dataloader), acc))
    
    return running_loss/len(dataloader), acc


def inference(model,
              dataloader, 
              load_dir,
              task_id):
    
    total = 0
    correct = 0
    
    with open(load_dir + 'best_model.pkl', 'rb') as f:
        checkpoint = pickle.load(f)
        model.load_parameters(checkpoint['parameters'])

    for X_batch, y_batch in dataloader:
        total += X_batch.shape[0]
        X_batch = batch_norm(X_batch)
        output, _ = model(X_batch)
        prediction = np.argmax(output, axis=1)
        label = np.argmax(y_batch, axis=1)
        correct += np.sum(prediction == label)
        
    acc = 100. * correct / total
    print("Task_id:{}, Inference: Acc:{}%".format(task_id, acc))

    return acc
