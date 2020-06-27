from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import gluOrtho2D
from typing import Tuple, List

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

import time

window = 0
# width, height = 600, 900
width, height = 800, 800

alpha = 0.1
# alpha = 0.01
# theta = np.zeros((3, 1))
theta = np.random.randn(3, 1)
bounds = (-5, 5, -5, 5)
accuracy = -1
k = 0
max_iters = 300
finished = False


def refresh2d(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # gluOrtho2D(-5, 12, -5, 12)
    gluOrtho2D(*bounds)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def line(x1, y1, x2, y2):
    glBegin(GL_LINES)
    glVertex2f(x1, y1)
    glVertex2f(x2, y2)
    glEnd()


def circle(radius: float, center: Tuple[int, int], side_num=360):
    glBegin(GL_TRIANGLE_FAN)

    x, y = center
    for vertex in range(0, side_num):
        angle = float(vertex) * 2.0 * np.pi / side_num
        glVertex2f(np.cos(angle) * radius + x, np.sin(angle) * radius + y)

    glEnd()


def sigmoid(z):
    return 1/(1+np.exp(-z))


def d_sigmoid(phi_x, y_hat, Y):
    return 1/len(phi_x)*phi_x.T@(y_hat - Y[:, np.newaxis])


def draw():
    global theta, accuracy, finished, k

    glClearColor(1, 1, 1, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    refresh2d(width, height)

    for x, y in zip(X, Y):
        if y:
            # y=1 amarillo
            glColor3f(0.99215686, 0.90588235, 0.14509804)
        else:
            # y = 0 lila
            glColor3f(0.26666667, 0.00392157, 0.32941176)
        circle(radius=0.07, center=(x[0], x[1]))

    try:
        x_min = bounds[0]
        x_max = bounds[1]
        y_min = int(-(theta[0] + theta[1] * x_min) / theta[2])
        y_max = int(-(theta[0] + theta[1] * x_max) / theta[2])
        glColor3f(1, 0, 0)
        line(x_min, y_min, x_max, y_max)
    except ValueError:
        pass

    # Logistic Regression
    z = phi_x@theta
    y_hat = sigmoid(z)

    grad = d_sigmoid(phi_x, y_hat, Y)
    if np.abs(grad.max()) > 1e-4 and accuracy <= 0.99 and k < max_iters:
        theta = theta - alpha*grad
        accuracy = accuracy_score(y_hat > 0.5, Y)
        k += 1
    else:
        if not finished:
            print(accuracy, grad.max())
            finished = True
    # time.sleep(0.05)
    glFlush()


if __name__ == '__main__':
    # X, Y = np.load('../x.npy'), np.load('../y.npy')
    X, Y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)
    X = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)
    phi_x = np.hstack((np.ones((len(X), 1)), X))

    glutInit()
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
    glutInitWindowSize(width, height)

    glutInitWindowPosition(200, 200)

    window = glutCreateWindow("2D Logistic Regression")

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glutDisplayFunc(draw)
    glutIdleFunc(draw)

    glutMainLoop()
