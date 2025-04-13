class my_SGD:
    def __init__(self, parameters, learning_rate, decay=0.5, weight_decay=0.0):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.decay = decay
        self.weight_decay = weight_decay
        
    def step(self, grads):
        for l in range(3):
            # L2正则化项：+ weight_decay * W
            self.parameters["W" + str(l + 1)] -= self.learning_rate * (
                grads["dW" + str(l + 1)] + self.weight_decay * self.parameters["W" + str(l + 1)]
            )
            self.parameters["b" + str(l + 1)] -= self.learning_rate * grads["db" + str(l + 1)]  # 一般不对bias做正则化
    
    def adjust_learning_rate(self):
        self.learning_rate *= self.decay

