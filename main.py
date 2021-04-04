#!/usr/bin/python3

import sys
import utils
from snake import Snake
import numpy as np

if len(sys.argv) != 10:
    print("Usage: " + sys.argv[0] + " input_image initial_snake output_image alpha beta tau w_line w_edge kappa")
    sys.exit(1)

input_image   = sys.argv[1]
initial_snake = sys.argv[2]
output_image  = sys.argv[3]
alpha         = float(sys.argv[4])
beta          = float(sys.argv[5])
tau           = float(sys.argv[6])
w_line        = float(sys.argv[7])
w_edge        = float(sys.argv[8])
kappa         = float(sys.argv[9])


snake = Snake(initial_snake)
snake.set_external_forces(input_image, balloon=kappa, k=1.0, w_line=w_line, w_edge=w_edge)
snake.set_parameters(alpha=alpha, beta=beta, tau=tau)
snake.go()
utils.display_snake(snake.img, np.loadtxt(initial_snake), snake.curve)
