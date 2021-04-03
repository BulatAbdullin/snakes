import numpy as np
import skimage.io as skio
import skimage.filters as skfilters
import scipy.sparse as sp

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
        A = sp.diags(diagonals, offsets, shape=(self.n, self.n)).toarray()
        self.iter_matrix = np.linalg.inv(np.eye(self.n) + self.tau*A)


    def set_external_forces(self, img_fname, k=0.1, w_line=0.0, w_edge=1.0, sigma=1.0):
        self.img = skio.imread(img_fname)

        potential_line = -skfilters.gaussian(self.img, sigma=sigma)
        potential_edge = -skfilters.sobel(potential_line)**2
        potential = -(w_line*potential_line + w_edge*potential_edge)

        # force consists of 'x' and 'y' components
        force = -np.array([skfilters.sobel_v(self.img), skfilters.sobel_h(self.img)])

        # normalize the forces
        norms = np.linalg.norm(force, axis=0)
        # hack to avoid division by zero
        norms[np.where(norms == 0)] = 1.0
        self.external_force = force / norms
        return k * force


    def update_curve(self):
        fx = Snake.bilinear_interpolate(self.external_force[0], self.curve[:, 0], self.curve[:, 1])
        fy = Snake.bilinear_interpolate(self.external_force[1], self.curve[:, 0], self.curve[:, 1])
        f_ext = np.hstack((fx[:, np.newaxis], fy[:, np.newaxis]))
        new_curve = self.iter_matrix @ (self.curve + self.tau*f_ext)
        curve_diff = np.linalg.norm(new_curve - self.curve)
        self.curve = new_curve
        return curve_diff


    def go(self, eps):
        curve_diff = self.update_curve()
        while curve_diff > eps:
            curve_diff = self.update_curve()


    @staticmethod
    def bilinear_interpolate(im, x_in, y_in):
        x = np.copy(x_in)
        y = np.copy(y_in)

        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, im.shape[1]-1);
        x1 = np.clip(x1, 0, im.shape[1]-1);
        y0 = np.clip(y0, 0, im.shape[0]-1);
        y1 = np.clip(y1, 0, im.shape[0]-1);

        Ia = im[ y0, x0 ]
        Ib = im[ y1, x0 ]
        Ic = im[ y0, x1 ]
        Id = im[ y1, x1 ]

        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)

        return wa*Ia + wb*Ib + wc*Ic + wd*Id
