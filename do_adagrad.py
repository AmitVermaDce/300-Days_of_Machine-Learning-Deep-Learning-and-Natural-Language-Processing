import numpy as np


def sigmoid(w, b, x, ):
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


def do_adagrad(X, Y):
    w, b, eta, max_epochs = -2, -2, 1.0, 1000
    v_w, v_b, eps = 0, 0, 1e-8
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
        v_w = v_w + dw ** 2
        v_b = v_b + db ** 2
        w = w - (eta / (np.sqrt(v_w + eps))) * dw
        b = b - (eta / np.sqrt(v_b + eps)) * db

    return w, b
