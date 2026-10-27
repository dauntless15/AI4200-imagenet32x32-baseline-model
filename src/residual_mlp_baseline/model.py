import torch
import torch.nn as nn



TASKS = {
    "mnist":      {"in_features": 28 * 28,     "num_classes": 10},    # 28x28 grayscale
    "cifar10":    {"in_features": 3 * 32 * 32, "num_classes": 10},    # 32x32 color
    "imagenet32": {"in_features": 3 * 32 * 32, "num_classes": 1000},  # 32x32 color
}


def activation_function(name):
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return torch.relu
    if name == "gelu":
        return torch.nn.functional.gelu
    raise ValueError(f"Unknown activation {name!r}; expected tanh, relu, or gelu")


def normalization_layer(name, width):
    if name == "layernorm":
        return nn.LayerNorm(width)
    if name == "batchnorm":
        return nn.BatchNorm1d(width)
    raise ValueError(
        f"Unknown normalization {name!r}; expected layernorm or batchnorm"
    )


class Block(nn.Module):
    
    def __init__(
        self, width, hidden, activation="tanh", normalization="layernorm"
    ):
        super().__init__()
        self.norm = normalization_layer(normalization, width)
        self.fc1 = nn.Linear(width, hidden)   # expand:  width  -> hidden
        self.fc2 = nn.Linear(hidden, width)   # project: hidden -> width
        self.activation = activation_function(activation)

    def forward(self, x):
        h = self.fc2(self.activation(self.fc1(self.norm(x))))
        return x + h                          # skip connection: add the input back


class ResidualMLP(nn.Module):
    

    def __init__(
        self,
        in_features,
        num_classes,
        width=128,
        hidden=256,
        depth=30,
        activation="tanh",
        normalization="layernorm",
    ):
        super().__init__()
        self.activation_name = activation
        self.activation = activation_function(activation)
        self.normalization_name = normalization

        self.input_layer = nn.Linear(in_features, width)

        self.blocks = nn.ModuleList([
            Block(
                width,
                hidden,
                activation=activation,
                normalization=normalization,
            )
            for _ in range(depth)
        ])

        self.head_norm = normalization_layer(normalization, width)
        self.head_fc1 = nn.Linear(width, hidden)
        self.head_fc2 = nn.Linear(hidden, num_classes)

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = x.flatten(start_dim=1)          
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.head_fc2(self.activation(self.head_fc1(self.head_norm(x))))
        return x

    def predict(self, x):
        self.eval()                           
        with torch.no_grad():                 
            return torch.softmax(self.forward(x), dim=1)
