import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class ANN:
    def __init__(self):
        self.weights_input_hidden = np.array([
            [0.20330057, 2.40007588, 0.20330057, 2.40007588, 0.20330057, 2.40007588],
            [0.2297051,  2.40068288, 0.2297051,  2.40068288, 0.2297051,  2.40068288],
            [0.21320227, 2.4003035,  0.21320227, 2.4003035,  0.21320227, 2.4003035 ],
            [0.22640453, 2.400607,   0.22640453, 2.400607,   0.22640453, 2.400607  ]
        ])  

        self.weights_hidden_output = np.array([
            [0.64774128],
            [0.73456161],
            [0.64774128],
            [0.73456161],
            [0.64774128],
            [0.73456161]
        ]) 
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden)  
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) 
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y, learning_rate=0.1):
        output_error = y - self.final_output
        output_delta = output_error * sigmoid_derivative(self.final_output)
        
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            error = (y - self.final_output)
            print(f"Epoch {epoch}, Error: {error}")
       

        print("\nFinal weights from input to hidden layer:\n", self.weights_input_hidden)
        print("Final weights from hidden to output layer:\n", self.weights_hidden_output)


X = np.array([[0.1, 0.9, 0.4, 0.8]])  

y = np.array([[1]])
ann = ANN()
ann.train(X, y, epochs=300000, learning_rate=0.1) 

predictions = ann.forward(X)
print("\nPrediction after training:")
print(predictions)  
