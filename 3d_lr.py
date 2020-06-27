# Rafael Villca Poggian

from OpenGL.GL import *
from OpenGL.GLUT import *

from ctypes import c_float, c_short

from Quaternion import *

from sklearn.datasets import make_blobs, make_classification
from sklearn.metrics import accuracy_score
from matplotlib.tri import Triangulation

import time

window = 0
width, height = 800, 830

mouse_x, mouse_y = 0, 0

x_angle = 0
y_angle = 0
z_angle = 0

h_state = 0
v_state = 0

h_increment = 0
v_increment = 0

clicked = False


alpha = 0.03
theta = np.random.randn(4, 1)
bounds = (-5, 5, -5, 5)
accuracy = -1
k = 0
max_iters = 500
finished = False


q = np.array([0, 0, 0, 1])
quaternion_matrix = toMatrix(np.array([0, 0, 0, 1]))
normals = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])


def refresh2d(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    limit = 3
    if width <= height:
        glOrtho(-limit, limit, -limit * height / width, limit * height / width, -10, 10)
    else:
        glOrtho(-limit * width / height, limit * width / height, -limit, limit, -10, 10)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glShadeModel(GL_FLAT)
    glEnable(GL_DEPTH_TEST)


def draw_translate(drawable, translate_vector):
    glPushMatrix()
    glTranslate(*translate_vector)
    drawable.draw()
    glPopMatrix()


def sigmoid(z):
    return 1/(1+np.exp(-z))


def d_sigmoid(phi_x, y_hat, Y):
    return 1/len(phi_x)*phi_x.T@(y_hat - Y[:, np.newaxis])


def triangulate(x, y):
    tri = Triangulation(x, y)
    return tri.get_masked_triangles()


def draw():
    global x_angle, y_angle, z_angle
    global theta, accuracy, finished, k

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    refresh2d(width, height)

    if h_increment != 0:
        rotate(h_increment, [0, 1, 0])
    if v_increment != 0:
        rotate(v_increment, [1, 0, 0])

    glMultMatrixf(quaternion_matrix)

    """Drawing"""
    for i, (x, y) in enumerate(zip(X, Y)):
        if y:
            # y=1 amarillo
            glColor3f(0.99215686, 0.90588235, 0.14509804)
        else:
            # y = 0 lila
            glColor3f(0.26666667, 0.00392157, 0.32941176)
        glPushMatrix()
        glTranslate(x[0], x[1], x[2])
        glutSolidSphere(0.07, 50, 50)
        glPopMatrix()

    plane_y = -(theta[0] + theta[1] * X[:, 0] + theta[2] * X[:, 1]) / theta[3]
    glColor(1, 0, 0)
    glBegin(GL_TRIANGLES)
    for triangle in triangles:
        x = X[triangle]
        y = plane_y[triangle]
        for xi, yi in zip(x, y):
            glVertex3f(xi[0], xi[1], yi)
    glEnd()

    z = phi_x @ theta
    y_hat = sigmoid(z)

    grad = d_sigmoid(phi_x, y_hat, Y)
    if np.abs(grad.max()) > 1e-4 and accuracy <= 0.99 and k < max_iters:
        theta = theta - alpha * grad
        accuracy = accuracy_score(y_hat > 0.5, Y)
        k += 1
    else:
        if not finished:
            print(accuracy, grad.max())
            finished = True

    glFlush()

    x_angle = (x_angle + v_increment) % 360
    y_angle = (y_angle + h_increment) % 360
    z_angle = 0


def hold_movement(mx, my):
    global clicked, h_state, v_state, h_increment, v_increment

    if clicked:
        h_increment = (mx - h_state) * 0.4
        h_state = mx

        v_increment = (my - v_state) * 0.4
        v_state = my
    else:
        h_increment = 0
        v_increment = 0


def click_control(button, state, x, y):
    global clicked, h_state, v_state, h_increment, v_increment

    if state == GLUT_DOWN and button == GLUT_LEFT_BUTTON:
        clicked = (clicked + 1) % 2
        h_state = x
        v_state = y

        if not clicked:
            h_increment = 0
            v_increment = 0


def rotate(degrees, axis):
    global quaternion_matrix, normals, q

    r = quaternion(np.array(axis), degrees)
    normals[0] = encase3(normals[0], r)
    normals[1] = encase3(normals[1], r)
    normals[2] = encase3(normals[2], r)
    q = quatMult(q, quaternion(axis, -degrees))
    quaternion_matrix = toMatrix(q)


if __name__ == '__main__':
    X, Y = make_classification(n_samples=100, n_features=3, n_redundant=0, n_informative=1, n_clusters_per_class=1)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    phi_x = np.hstack((np.ones((len(X), 1)), X))
    triangles = triangulate(X[:, 0], X[:, 1])

    glutInit()
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutInitWindowPosition(0, 0)
    window = glutCreateWindow("3D Logistic Regression")
    glutDisplayFunc(draw)
    glutIdleFunc(draw)

    glutMouseFunc(click_control)

    glutPassiveMotionFunc(hold_movement)

    glutMainLoop()
