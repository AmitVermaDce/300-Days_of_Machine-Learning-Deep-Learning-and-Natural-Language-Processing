import numpy as np
import math


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


def do_adam(X, Y):
    w, b, eta, max_epochs = -2, -2, 0.1, 1000
    m_w, m_b, v_w, v_b, beta1, beta2, eps = 0, 0, 0, 0, 0.9, 0.999, 1e-8
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
        m_w = (m_w * beta1) + (1 - beta1) * dw
        m_b = (m_b * beta1) + (1 - beta1) * db

        v_w = (beta2 * v_w) + (1 - beta2) * (dw ** 2)
        v_b = (beta2 * v_b) + (1 - beta2) * (db ** 2)

        m_w = m_w / (1 - math.pow(beta1, i + 1))
        m_b = m_b / (1 - math.pow(beta1, i + 1))
        v_w = v_w / (1 - math.pow(beta2, i + 1))
        v_b = v_b / (1 - math.pow(beta2, i + 1))

        w = w - (eta / np.sqrt(v_w + eps)) * m_w
        b = b - (eta / np.sqrt(v_b + eps)) * m_b
    return w, b
