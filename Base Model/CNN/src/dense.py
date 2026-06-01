import numpy as np


class Dense:
    """
    Fully Connected Layer

    Parameters
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of neurons.

    activation : str | None
        relu
        sigmoid
        softmax
        None
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = None
    ):

        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation

        self.weights = None
        self.bias = None

        self.initialize()

    # ==================================================
    # INITIALIZATION
    # ==================================================

    def initialize(self):
        """
        Xavier initialization.
        """

        limit = np.sqrt(6 / (self.in_features + self.out_features))

        self.weights = np.random.uniform(
            -limit,
            limit,
            (self.out_features, self.in_features)
        )

        self.bias = np.zeros((self.out_features, 1))

    # ==================================================
    # LINEAR
    # ==================================================

    def linear(self, x):
        """
        z = Wx + b

        x shape:
            (in_features, 1)

        return:
            (out_features, 1)
        """

        return np.dot(self.weights, x) + self.bias

    # ==================================================
    # ACTIVATIONS
    # ==================================================

    def relu(self, z):

        return np.maximum(0, z)

    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))

    def softmax(self, z):

        z = z - np.max(z)

        exp_z = np.exp(z)

        return exp_z / np.sum(exp_z)

    # ==================================================
    # ACTIVATION SELECTOR
    # ==================================================

    def activation(self, z):

        if self.activation_name is None:
            return z

        if self.activation_name == "relu":
            return self.relu(z)

        if self.activation_name == "sigmoid":
            return self.sigmoid(z)

        if self.activation_name == "softmax":
            return self.softmax(z)

        raise ValueError(
            f"Unsupported activation: {self.activation_name}"
        )

    # ==================================================
    # FORWARD
    # ==================================================

    def forward(self, x):
        """
        Forward pass.

        Input:
            (in_features,)
            or
            (in_features,1)

        Output:
            (out_features,1)
        """

        x = np.asarray(x)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        z = self.linear(x)

        output = self.activation(z)

        return output