# A tiny neural-net + backprop library (NumPy only), with SGD optimizer.
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


np.set_printoptions(precision=6, suppress=True)


def _as_array(x):
    return np.asarray(x, dtype=np.float64)


@dataclass
class Parameter:
    data: np.ndarray
    grad: np.ndarray
    name: str = ""

    @staticmethod
    def from_shape(shape, name=""):
        # He/Xavier-ish scaling for stable starts
        scale = 1.0 / np.sqrt(shape[1] if len(shape) > 1 else max(1, shape[0]))
        data = np.random.randn(*shape) * scale
        return Parameter(data=_as_array(data), grad=np.zeros_like(_as_array(data)), name=name)


class Module:
    def parameters(self) -> List[Parameter]:
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad[...] = 0.0


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool=True, name: str="linear"):
        self.W = Parameter.from_shape((out_features, in_features), name=f"{name}.weight")
        self.b = Parameter.from_shape((out_features,), name=f"{name}.bias") if bias else None
        self._x_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = _as_array(x)
        self._x_cache = x
        y = x @ self.W.data.T
        if self.b is not None:
            y = y + self.b.data
        return y

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x = self._x_cache
        self.W.grad += grad_out.T @ x
        if self.b is not None:
            self.b.grad += np.sum(grad_out, axis=0)
        dx = grad_out @ self.W.data
        return dx

    def parameters(self) -> List[Parameter]:
        return [p for p in [self.W, self.b] if p is not None]


class ReLU(Module):
    def __init__(self):
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = _as_array(x)
        self._mask = x > 0
        return np.maximum(x, 0.0)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self._mask


class Sigmoid(Module):
    def __init__(self):
        self._out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = _as_array(x)
        out = 1.0 / (1.0 + np.exp(-x))
        self._out = out
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        out = self._out
        return grad_out * out * (1.0 - out)

class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            if hasattr(layer, "backward"):
                grad_out = layer.backward(grad_out)
        return grad_out

    def parameters(self) -> List[Parameter]:
        ps = []
        for layer in self.layers:
            ps.extend(layer.parameters())
        return ps


class MSELoss:
    def __init__(self):
        self._y_pred = None
        self._y_true = None
        self._N = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = _as_array(y_pred)
        y_true = _as_array(y_true)
        self._y_pred, self._y_true = y_pred, y_true
        self._N = y_pred.size
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self) -> np.ndarray:
        return (2.0 / self._N) * (self._y_pred - self._y_true)


class SGD:
    def __init__(self, params: List[Parameter], lr=1e-2, momentum: float=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self._velocity = {id(p): np.zeros_like(p.data) for p in params}

    def step(self):
        for p in self.params:
            v = self._velocity[id(p)]
            if self.momentum != 0.0:
                v[...] = self.momentum * v - self.lr * p.grad
                p.data[...] = p.data + v
            else:
                p.data[...] = p.data - self.lr * p.grad


def finite_diff_grad(model: Module, loss_fn: MSELoss, x, y, eps=1e-6):
    # Finite-difference gradients for verification.
    y_pred = model.forward(x)
    base_loss = loss_fn.forward(y_pred, y)
    grads_fd = {}

    for p in model.parameters():
        g_fd = np.zeros_like(p.data)
        it = np.nditer(p.data, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old = p.data[idx]
            p.data[idx] = old + eps
            lp = loss_fn.forward(model.forward(x), y)
            p.data[idx] = old - eps
            lm = loss_fn.forward(model.forward(x), y)
            p.data[idx] = old
            g_fd[idx] = (lp - lm) / (2*eps)
            it.iternext()
        grads_fd[p.name] = g_fd
    return grads_fd, base_loss
