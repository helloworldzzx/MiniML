import numpy as np

def summary(model, input_shape):

    x = np.zeros((1, *input_shape))

    print("\nModel Summary")
    print("=" * 50)

    total_params = 0

    for layer in model.layers:

        x = layer.forward(x)

        params = 0
        if hasattr(layer, "parameters"):
            for p in layer.parameters():
                params += p.data.size

        total_params += params

        print(f"{layer.__class__.__name__}")
        print(f" output shape: {x.shape}")
        print(f" params: {params}")
        print("-" * 50)

    print("Total params:", total_params)