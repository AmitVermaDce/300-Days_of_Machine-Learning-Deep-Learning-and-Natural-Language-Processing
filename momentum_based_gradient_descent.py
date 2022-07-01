import numpy as np


def sigmoid(w, b, x):
    return 1.0 / (1 + np.exp(-1 * (w * x + b)))


def error(w, b, X, Y):
    err = 0.0
    for x, y in zip(X, Y):
        fx = sigmoid(w, b, x)
        err += 0.5 * (fx - y) ** 2
    return err


def grad_w(w, b, x, y):
    fx = sigmoid(w, b, x)
    return (fx - y) * fx * (1 - fx) * x


def grad_b(w, b, x, y):
    fx = sigmoid(w, b, x)
    return (fx - y) * fx * (1 - fx)


def do_momentum_based_gradient_descent(X, Y):
    w, b, eta, max_epochs = -2, -2, 1.0, 1000
    prev_v_w, prev_v_b, gamma = 0, 0, 0.9
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
        v_w = (gamma * prev_v_w) + (eta * dw)  # adding momentum
        v_b = (gamma * prev_v_b) + (eta * db)  # adding momentum
        w = w - v_w
        b = b - v_b
        prev_v_w = v_w
        prev_v_b = v_b

    return w, b
