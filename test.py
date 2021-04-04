import utils
import numpy as np
import skimage.io as skio
from snake import Snake

def display_external_forces():
    snake = Snake("../testdata/astranaut_init_snake.txt")
    snake.set_external_forces("../testdata/astranaut.png")


def display_snake():
    snake = Snake("../testdata/astranaut_init_snake.txt")
    snake.set_external_forces("../testdata/astranaut.png", balloon=-0.4, k=1.0, w_line=0, w_edge=1.0)
    h = snake.h
    snake.set_parameters(alpha=1e-5, beta=1e-7, tau=1)
    snake.go()
    utils.display_snake(snake.img, np.loadtxt("../testdata/astranaut_init_snake.txt"), snake.curve)


def setup():
    snake = Snake("../testdata/i.txt")
    snake.set_parameters(alpha=1.0, beta=2.0, tau=0.1)
    snake.setup_iteration_matrix()
    print(snake.iter_matrix)
    # snake.set_external_forces("../testdata/astranaut.png")
    # snake.update_curve()
    # print(snake.curve)
