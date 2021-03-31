import numpy as np
import skimage.io as skio
import skimage.filters as skfilters
import scipy.sparse as sp
from scipy.sparse.linalg import inv

class Snake:
    def __init__(self, initial_approximation_fname):
        # throw away the last point because it coincides with the first point
        self.curve = np.loadtxt(initial_approximation_fname)[: -1]

        self.n = self.curve.shape[0] # number of points
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
        self.iter_matrix = (A.tocsc() + self.tau*sp.eye(self.n)).todense().I


    def set_external_forces(self, img_fname, k=0.1, w_line=0.0, w_edge=1.0, sigma=1.0):
        img = skio.imread(img_fname)
        self.external_force = Snake.external_force(img, k, w_line, w_edge, sigma)


    def update_curve(self):
        f_ext = self.external_force[:, self.curve[:, 0].astype(int), self.curve[:, 1].astype(int)].T
        new_curve = self.iter_matrix @ (self.curve + self.tau*f_ext)
        curve_diff = np.linalg.norm(new_curve - self.curve)
        self.curve = new_curve
        return curve_diff


    @staticmethod
    def external_force(img, k, w_line, w_edge, sigma):
        potential_line = -skfilters.gaussian(img, sigma=sigma)
        potential_edge = -skfilters.sobel(potential_line)**2
        potential = -(w_line*potential_line + w_edge*potential_edge)

        # force consists of 'x' and 'y' components
        force = -np.array([skfilters.sobel_v(img), skfilters.sobel_h(img)])

        # normalize the forces
        norms = np.linalg.norm(force, axis=0)
        # hack to avoid division by zero
        norms[np.where(norms == 0)] = 1.0
        force /= norms
        return k * force
