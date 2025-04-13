import numpy as np
from utils import *

# class MyModel:
#     def __init__(self, in_channel, hidden_layer, num_classes, activation='relu'):
#         self.in_channel= in_channel
#         self.n_hidden_1 = hidden_layer[0]
#         self.n_hidden_2 = hidden_layer[1]
#         self.num_classes = num_classes
        
#         self.W1 = np.random.normal(0,np.sqrt(2/ self.in_channel),(self.in_channel, self.n_hidden_1))
#         self.b1 = np.zeros((1, self.n_hidden_1))
        
#         self.W2 = np.random.normal(0,np.sqrt(2/self.n_hidden_1),(self.n_hidden_1, self.n_hidden_2))
#         self.b2 = np.zeros((1, self.n_hidden_2))
        
#         self.W3 = np.random.normal(0,np.sqrt(2/self.n_hidden_2),(self.n_hidden_2, self.num_classes))
#         self.b3 = np.zeros((1, self.num_classes))
        
#         self.parameters = {'W1':self.W1,'b1':self.b1,
#                      'W2':self.W2,'b2':self.b2,
#                      'W3':self.W3,'b3':self.b3
#                      }
        
#         self.relu = relu
#         self.softmax = softmax
        
#     def forward(self,x):
#         z1 = np.dot(x, self.W1) + self.b1
#         a1 = self.relu(z1)
        
#         z2 = np.dot(a1, self.W2) + self.b2
#         a2 = self.relu(z2)
 
#         z3 = np.dot(a2, self.W3) + self.b3
#         a3 = self.softmax(z3)
        
#         cache = (z1, a1, z2, a2, z3, a3)
#         return (a3, cache)
    
#     def backward(self,x,y,cache):
#         (z1, a1, z2, a2, z3, a3) = cache
#         bs = x.shape[0]
#         dz3 = 1.0 / bs * (a3 - y) # B x Classes
#         dw3 = a2.T @ dz3 # H2 x classes
#         db3 = np.sum(dz3,axis=0).T # Classes x 1
        
#         da2 = dz3 @ self.W3.T # B x H2
#         dz2 = np.multiply(da2, np.int64(a2 > 0)) # B x H2
#         dw2 = a1.T @ dz2 # H1 x H2
#         db2 = np.sum(dz2, axis=0).T # H2 x 1
        
#         da1 = dz2 @ self.W2.T # B x H1
#         dz1 = np.multiply(da1, np.int64(a1 > 0)) # B x H1
#         dw1 = x.T @ dz1 # D x H1
#         db1 = np.sum(dz1,axis=0).T # B x H1
        
#         gradients = {"dW3": dw3, "db3": db3,
#                 "dW2": dw2, "db2": db2,
#                 "dW1": dw1, "db1": db1}
    
        
#         return gradients
        
    
#     def load_parameters(self, para):
#         self.W1 = para['W1']
#         self.b1 = para['b1']
#         self.W2 = para['W2']
#         self.b2 = para['b2']
#         self.W3 = para['W3']
#         self.b3 = para['b3']

#     def __call__(self, x):
#         return self.forward(x)

class MyModel:
    def __init__(self, in_channel, hidden_layer, num_classes, activation='relu'):
        self.in_channel = in_channel
        self.n_hidden_1 = hidden_layer[0]
        self.n_hidden_2 = hidden_layer[1]
        self.num_classes = num_classes
        
        self.W1 = np.random.normal(0, np.sqrt(2 / self.in_channel), (self.in_channel, self.n_hidden_1))
        self.b1 = np.zeros((1, self.n_hidden_1))
        
        self.W2 = np.random.normal(0, np.sqrt(2 / self.n_hidden_1), (self.n_hidden_1, self.n_hidden_2))
        self.b2 = np.zeros((1, self.n_hidden_2))
        
        self.W3 = np.random.normal(0, np.sqrt(2 / self.n_hidden_2), (self.n_hidden_2, self.num_classes))
        self.b3 = np.zeros((1, self.num_classes))
        
        self.parameters = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3
        }

        # 选择激活函数和导数
        if activation == 'relu':
            self.activation_func = relu
            self.activation_deriv = relu_derivative
        elif activation == 'sigmoid':
            self.activation_func = sigmoid
            self.activation_deriv = sigmoid_derivative
        elif activation == 'tanh':
            self.activation_func = tanh
            self.activation_deriv = tanh_derivative
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.softmax = softmax

    def forward(self, x):
        z1 = np.dot(x, self.W1) + self.b1
        a1 = self.activation_func(z1)
        
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.activation_func(z2)
 
        z3 = np.dot(a2, self.W3) + self.b3
        a3 = self.softmax(z3)
        
        cache = (x, z1, a1, z2, a2, z3, a3)
        return a3, cache

    def backward(self, y, cache):
        x, z1, a1, z2, a2, z3, a3 = cache
        bs = x.shape[0]

        dz3 = (a3 - y) / bs
        dw3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * self.activation_deriv(z2)
        dw2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.activation_deriv(z1)
        dw1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        gradients = {
            "dW3": dw3, "db3": db3,
            "dW2": dw2, "db2": db2,
            "dW1": dw1, "db1": db1
        }

        return gradients

    def load_parameters(self, para):
        self.W1 = para['W1']
        self.b1 = para['b1']
        self.W2 = para['W2']
        self.b2 = para['b2']
        self.W3 = para['W3']
        self.b3 = para['b3']

    def __call__(self, x):
        return self.forward(x)
    