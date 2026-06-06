"""
dense.py

CNN From Scratch

Fully Connected Layer

Forward:
    z = Wx + b

Backward:
    dW
    db
    dX
"""

from __future__ import annotations

import numpy as np


class Dense:
    """
    Fully Connected Layer

    Input:
        (input_dim,)

    Output:
        (output_dim,)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        random_seed: int | None = None
    ) -> None:

        if random_seed is not None:
            np.random.seed(random_seed)

        self.input_dim = input_dim
        self.output_dim = output_dim

        # He Initialization
        scale = np.sqrt(2.0 / input_dim)

        self.weights = (
            np.random.randn(
                output_dim,
                input_dim
            )
            * scale
        )

        self.bias = np.zeros(
            output_dim,
            dtype=np.float64
        )

        # Cache
        self.input_cache = None
        self.output_cache = None
        
        # For backward visualization 
        self.grad_input_cache = None

        # Gradients
        self.dW = np.zeros_like(
            self.weights
        )

        self.db = np.zeros_like(
            self.bias
        )

    # Forward

    def forward(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Forward Pass

        z = Wx + b

        Parameters
        ----------
        x : np.ndarray
            Shape:
                (input_dim,)

        Returns
        -------
        np.ndarray
            Shape:
                (output_dim,)
        """

        x = np.asarray(x, dtype=np.float64)

        if x.ndim != 1:
            raise ValueError(
                "Dense expects a 1D vector."
            )

        if x.shape[0] != self.input_dim:
            raise ValueError(
                f"Expected input size "
                f"{self.input_dim}, "
                f"received {x.shape[0]}"
            )

        self.input_cache = x

        output = (
            self.weights @ x
            + self.bias
        )

        self.output_cache = output

        return output

    # Backward

    def backward(
        self,
        grad_output: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradients.

        dW = dZ * x^T
        db = dZ
        dX = W^T * dZ
        
        Parameters 
        ---------- 
        grad_output : np.ndarray 
            dL/dZ 
            
            Shape: 
                (output_dim,) 
        
        Returns 
        ------- 
        np.ndarray 
            dL/dX 
            
            Shape: 
                (input_dim,)
        """

        if self.input_cache is None:
            raise RuntimeError(
                "forward() must be called "
                "before backward()."
            )

        grad_output = np.asarray(
            grad_output,
            dtype=np.float64
        )
        
        if grad_output.ndim != 1: 
            raise ValueError( "grad_output must be 1D." )
        
        if grad_output.shape[0] != self.output_dim: 
            raise ValueError( 
                f"Expected gradient size " 
                f"{self.output_dim}, " 
                f"received " 
                f"{grad_output.shape[0]}" 
            )

        # --------------------------
        # dW
        # --------------------------
        self.dW = np.outer(
            grad_output,
            self.input_cache
        )

        # --------------------------
        # db
        # --------------------------
        self.db = grad_output.copy()

        # --------------------------
        # dX
        # --------------------------
        grad_input = (
            self.weights.T
            @ grad_output
        )
        
        self.grad_input_cache = ( 
            grad_input 
        )

        return grad_input

    # SGD Update

    def update(
        self,
        learning_rate: float
    ) -> None:
        """
        SGD

        W = W - lr * dW
        b = b - lr * db
        """

        self.weights -= (
            learning_rate
            * self.dW
        )

        self.bias -= (
            learning_rate
            * self.db
        )
        
    # Parameter API
    def parameters(self): 
        """ 
        Returns trainable parameters. 
        
        Useful for future optimizers. 
        """ 
        
        return [ 
            ( 
                self.weights, 
                self.dW 
            ), 
            ( 
                self.bias, 
                self.db 
            ) 
        ]

    # Utilities

    @property
    def num_parameters(self) -> int:

        return (
            self.weights.size
            + self.bias.size
        )

    def summary(self) -> None:

        print("\nDense")
        print("-" * 40)

        print(
            f"Input Dim  : {self.input_dim}"
        )

        print(
            f"Output Dim : {self.output_dim}"
        )

        print(
            f"Parameters : {self.num_parameters}"
        )

        print("-" * 40)

    def zero_grad(self) -> None:
        """
        Reset gradients.
        """

        self.dW.fill(0)
        self.db.fill(0)

    def __repr__(self) -> str:

        return (
            f"Dense("
            f"{self.input_dim}, "
            f"{self.output_dim}"
            f")"
        )