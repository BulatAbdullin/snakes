#!/usr/bin/python3

import sys

if len(sys.argv) != 10:
    print("Usage: " + sys.argv[0] + " input_image initial_snake output_image alpha beta tau w_line w_edge kappa")
    sys.exit(1)

import utils
from snake import Snake
import numpy as np

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
snake.set_parameters(alpha=alpha, beta=beta, tau=tau)
snake.set_external_forces(input_image, balloon=kappa, w_line=w_line, w_edge=w_edge)
snake.go(eps=0.001, n_max=400)
utils.save_mask(output_image, snake.curve, snake.img)
