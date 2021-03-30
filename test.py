import utils
import numpy as np
from skimage import io
from snake import Snake

def display_external_forces():
    img = io.imread("../testdata/astranaut.png")
    f_ext = Snake.external_force(img)

    snake = Snake("../testdata/astranaut_init_snake.txt")

    utils.display_image_in_actual_size(f_ext[0]**2 + f_ext[1]**2)


def setup():
    snake = Snake("../testdata/astranaut_init_snake.txt")
    snake.set_parameters(alpha=1.0, beta=2.0, tau=0.1)
