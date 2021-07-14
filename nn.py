import numpy as np


class Layer:
    """
    Bloque base para definición de una capa
    """
    def __init__(self):
        """Inicialización de parámetros."""
        pass
    
    def forward(self, input):
        """
        Recibe [batch, input_units], Y devuelve [batch, output_units]
        """
        return input

    def backward(self, input, grad_output):
        """
        Aplicamos Backpropagation
        
        d loss / d x  = (d loss / d layer) * (d layer / d x)
        
        Se recibe d loss / d layer como entrada, así que sólo se debe multiplicar por d layer / d x.
        
        Si la capa tiene parámetros, se deben actualizar sus valores d loss / d layer
        """
        num_units = input.shape[1]
        
        d_layer_d_input = np.eye(num_units)
        
        return np.dot(grad_output, d_layer_d_input)
        

class ReLU(Layer):
    def __init__(self):
        """No tiene parámetros entrenables"""
        pass
    
    def forward(self, input):
        """Aplica ReLU a [batch, input_units]"""
        return np.maximum(0,input)
    
    def backward(self, input, grad_output):
        """Computa gradiente de loss respecto a ReLU"""
        relu_grad = input > 0
        return grad_output*relu_grad    #dL/dZ[l]
        
        
class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1, initializer='standard'):
        """
        Una capa densa que realiza un mapeo entre espacios:
        f(x) = <W*x> + b
        """
        self.learning_rate = learning_rate
        
        ###PODEMOS ELEGIR LA FORMA DE INICIALIZACIÓN, LA ESTÁNDAR Y LA XAVIER###
        scaler = 0.01 if initializer == 'standard' else np.sqrt(2.0/input_units)
        
        self.weights = np.random.randn(input_units, output_units)*scaler
        self.biases = np.zeros(output_units)
        
    def forward(self,input):
        """
        input shape: [batch, input_units]
        output shape: [batch, output units]
        """
        return input @ self.weights + self.biases
    
    def backward(self,input,grad_output):
        # Calculamos dL/dA[l-1]
        grad_input = grad_output @ self.weights.T #dL/dA[l-1] = <dL/dZ[l], w[l]'> -> dL/dX -> dL/dA[l-1]
        
        # Calculamos para los pesos y los bias
        grad_weights = input.T @ grad_output #dL/dW[l] = <A[l-1]', dL/dZ[l]> -> dL/dθ
        grad_biases = np.sum(grad_output, axis=0) #dL/db[l] = sum_rows(b[l])
        
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        
        # Actualizamos parámetros
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input ## Retornamos grad para la anterior capa
        
        
def forward(network, X):
    """
    Damos una pasada por toda la red propagando la información hacia delante.
    """
    activations = []
    input = X

    for layer in network:
        input = layer.forward(input)
        activations.append(input)    
        
    assert len(activations) == len(network)
    return activations


def sigmoid(z):
    # Sigmoide estable numericamente
    return np.where(z >= 0, 1/(1 + np.exp(-z)), np.exp(z)/(1 + np.exp(z)))


def d_sigmoid(y_hat, Y):
    return 1/len(Y)*(y_hat - Y[:, np.newaxis])


def predict(network,X):
    """
    Calculamos un vector de predicciones
    """
    logits = forward(network,X)[-1]
    preds = sigmoid(logits)

    return (preds>0.5)*1


def step(network,X,y):
    """
    Realizar un paso de entrenamiento, un forward pass y un backward pass, junto con una actualización de pesos.
    """
    
    # Obtener activaciones
    layer_activations = forward(network,X)
    layer_inputs = [X]+layer_activations  #layer_input[i] es una entrada para network[i]
    logits = layer_activations[-1] #extraemos la última capa sin activar
    
    # Calculamos el Loss y el gradiente inicial para retropopagar
    loss_grad = d_sigmoid(sigmoid(logits), y) #dZ[L]
    
    # Retropopagamos
    for layer, input in zip(reversed(network), reversed(layer_inputs[:-1])):
        loss_grad = layer.backward(input, loss_grad)
