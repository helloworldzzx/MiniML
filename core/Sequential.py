from core.Module import Module

class Sequential(Module):
    def __init__(self,*layers):
        self.layers = list(layers)

    def forward(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self,grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(
                layer.parameters()
            )
        return params