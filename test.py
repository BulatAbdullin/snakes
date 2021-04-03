import utils
import numpy as np
import skimage.io as skio
from snake import Snake

def display_external_forces():
    img = skio.imread("../testdata/astranaut.png")
    f_ext = Snake.external_force(img)

    snake = Snake("../testdata/astranaut_init_snake.txt")

    utils.display_image_in_actual_size(f_ext[0]**2 + f_ext[1]**2)


def display_snake():
    snake = Snake("../testdata/astranaut_init_snake.txt")
    snake.set_external_forces("../testdata/astranaut.png")
    h = 1/399
    snake.set_parameters(alpha=h**2, beta=h**4, tau=1.0)
    snake.setup_iteration_matrix()
    for i in range(100):
        snake.update_curve()
    utils.display_snake(snake.img, np.loadtxt("../testdata/astranaut_init_snake.txt"), snake.curve)


def setup():
    snake = Snake("../testdata/i.txt")
    snake.set_parameters(alpha=1.0, beta=2.0, tau=0.1)
    snake.setup_iteration_matrix()
    print(snake.iter_matrix)
    # snake.set_external_forces("../testdata/astranaut.png")
    # snake.update_curve()
    # print(snake.curve)
