import numpy as np


def sigmoid(w, b, x, ):
    return 1.0 / (1 + np.exp(-1 * (w * x + b)))


def grad_w(w, b, x, y):
    fx = sigmoid(w, b, x)
    return (fx - y) * fx * (1 - fx) * x


def grad_b(w, b, x, y):
    fx = sigmoid(w, b, x)
    return (fx - y) * fx * (1 - fx)


def do_stochastic_nesterov_accelerated_gradient_descent(X, Y):
    w, b, eta, max_epochs = -2, -2, 1.0, 1000
    prev_v_w, prev_v_b, gamma = 0, 0, 0.9
    for i in range(max_epochs):
        dw, db = 0, 0
        # Partial updates for lookahead point
        v_w = gamma * prev_v_w
        v_b = gamma * prev_v_b

        # Calculate gradient after partial updates
        for x, y in range(X, Y):
            dw += grad_w(w - v_w, b - v_b, x, y)
            db += grad_b(w - v_w, b - v_b, x, y)
            v_w = (gamma * prev_v_w) + (eta * dw)
            v_b = (gamma * prev_v_b) + (eta * db)
            w = w - v_w
            b = b - v_b
            prev_v_w = v_w
            prev_v_b = v_b
    return w, b
