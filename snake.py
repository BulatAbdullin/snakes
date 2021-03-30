import numpy as np
from skimage import filters

class Snake:
    def __init__(self, initial_approximation_fname):
        self.curve = np.loadtxt(initial_approximation_fname)

        # subtract 1 because the first point coincides with the last point
        self.n = self.curve.shape[0] - 1 # number of points
        self.h = 1.0 / self.n


    def set_parameters(self, alpha, beta, tau):
        self.alpha = alpha
        self.beta = beta
        self.tau = tau


    def setup_band_matrix(self):
        pass


    def update(f_ext):
        pass


    @staticmethod
    def external_force(img, k=1.0, w_line=0.0, w_edge=1.0, sigma=1.0):
        potential_line = -filters.gaussian(img, sigma=sigma)
        potential_edge = -filters.sobel(potential_line)**2
        potential = -(w_line*potential_line + w_edge*potential_edge)

        # force consists of 'x' and 'y' components
        force = -np.array([filters.sobel_v(img), filters.sobel_h(img)])

        # normalize the forces
        norms = np.linalg.norm(force, axis=0)
        # hack to avoid division by zero
        norms[np.where(norms == 0)] = 1.0
        force /= norms
        return k * force
