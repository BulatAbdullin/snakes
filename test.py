import utils
from skimage import io
from snake import Snake

def display_external_forces():
    img = io.imread("../testdata/astranaut.png")
    f_ext = Snake.external_force(img)

    snake = Snake("../testdata/astranaut_init_snake.txt")

    utils.display_image_in_actual_size(f_ext[0]**2 + f_ext[1]**2)
