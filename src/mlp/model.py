import torch
import torch.nn as nn

class FlexibleMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation, num_classes=3, dropout=0.0, 
                 normalization_layers=[],   # capas (1,2,...) con Layer Normalization
                 dropout_layers=[]          # capas (1,2,...) con Dropout 
                ):
       
        super().__init__()
        activations = {
            'relu':    nn.ReLU,
            'tanh':    nn.Tanh,
            'gelu':    nn.GELU,
            'leakyrelu': nn.LeakyReLU,
            'elu':     nn.ELU,
            "silu":    nn.SiLU
        }
        
        act_fn = activations[activation.lower()]
        layers, prev_dim = [], input_dim

        for idx, h in enumerate(hidden_layers, start=1):
            # 1) Capa lineal
            layers.append(nn.Linear(prev_dim, h))

            # 2) Layer Normalization si idx está en norm_layers
            if idx in normalization_layers:
                layers.append(nn.LayerNorm(h)) 

            # 3) Activación
            layers.append(act_fn())

            # 4) Dropout si idx está en dropout_layers
            if dropout > 0 and idx in dropout_layers:
                layers.append(nn.Dropout(dropout))

            prev_dim = h

        # Capa de salida
        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)