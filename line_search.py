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


def do_gradient_descent_with_line_search(X, Y):
    w, b, etas, max_epochs = -2, -2, [0.1, 0.5, 1.0, 5.0, 10.0], 1000
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)

        best_w, best_b, min_error = w, b, 1000
        for eta in etas:
            temp_w = w - eta * dw
            temp_b = w - eta * db
            if error(temp_w, temp_b, X, Y) < min_error:
                best_w = temp_w
                best_b = temp_b
                min_error = error(temp_w, temp_b, X, Y)
        w, b = best_w, best_b

    return w, b
