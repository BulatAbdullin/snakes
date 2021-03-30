import numpy as np
from skimage import filters
import scipy.sparse as sp
from scipy.sparse.linalg import inv

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
        self.setup_iteration_matrix()


    def setup_iteration_matrix(self):
        a, b = self.alpha / self.h**2, self.beta / self.h**4
        diagonals = [-a - 4*b, b, b, -a - 4*b, 2*a + 6*b, -a - 4*b, b, b, -a - 4*b]
        offsets = [-self.n + 1, -self.n + 2, -2, -1, 0, 1, 2, self.n - 2, self.n - 1]
        A = sp.diags(diagonals, offsets, shape=(self.n, self.n))
        self.iter_matrix = inv(A.tocsc() + self.tau*sp.eye(self.n))


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
