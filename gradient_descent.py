import numpy as np


def sigmoid(w, b, x, ):
    return 1.0 / (1 + np.exp(-1 * (w * x + b)))


def grad_w(w, b, x, y):
    fx = sigmoid(w, b, x)
    return (fx - y) * fx * (1 - fx) * x


def grad_b(w, b, x, y):
    fx = sigmoid(w, b, x)
    return (fx - y) * fx * (1 - fx)


def do_gradient_descent(X, Y):
    w, b, eta, max_epochs = -2, -2, 1.0, 1000
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
        w = w - eta * dw
        b = b - eta * db

    return w, b
