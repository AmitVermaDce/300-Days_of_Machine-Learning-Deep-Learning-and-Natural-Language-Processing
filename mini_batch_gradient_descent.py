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


def do_mini_batch_gradient_descent(X, Y):
    w, b, eta, max_epochs = -2, -2, 1.0, 1000
    min_batch_size, num_points_seen = 2, 0
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
            num_points_seen += 1

            if num_points_seen % min_batch_size == 0:
                w = w - eta * dw
                b = b - eta * db
                dw, db = 0, 0  # reset the gradients
    return w, b
