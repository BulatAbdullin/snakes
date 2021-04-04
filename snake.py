import numpy as np
import skimage.io as skio
import skimage.filters as skfilters
import scipy.sparse as sp
from scipy.interpolate import interp1d

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


    def set_external_forces(self, img_fname, balloon=0.1, k=0.5, w_line=1.0, w_edge=2.0, sigma=1):
        self.img = skio.imread(img_fname)
        self.balloon = balloon

        potential_line = -skfilters.gaussian(self.img, sigma=sigma)
        potential_edge = -skfilters.sobel(potential_line)**2
        potential = -(w_line*potential_line + w_edge*potential_edge)

        # force consists of 'x' and 'y' components
        force = np.array([skfilters.sobel_v(self.img), skfilters.sobel_h(self.img)])

        # normalize the forces
        norms = np.linalg.norm(force, axis=0)
        # hack to avoid division by zero
        norms[np.where(norms == 0)] = 1.0
        self.external_force = force / norms
        self.external_force *= k


    def update_curve(self):
        fx = Snake.bilinear_interpolate(self.external_force[0], self.curve[:, 0], self.curve[:, 1])
        fy = Snake.bilinear_interpolate(self.external_force[1], self.curve[:, 0], self.curve[:, 1])
        f_ext = np.hstack((fx[:, np.newaxis], fy[:, np.newaxis]))

        derivative = 1/self.h * np.diff(self.curve, axis=0)
        derivative = np.append(derivative, derivative[0, np.newaxis], axis=0)
        # perpendicular to the curve
        balloon_force = derivative[:, [1, 0]]
        balloon_force[:, 1] *= -1
        # normalize balloon force
        balloon_force /= np.linalg.norm(balloon_force, axis=1)[:, np.newaxis]

        f_ext += self.balloon * balloon_force

        new_curve = self.iter_matrix @ (self.curve + self.tau*f_ext)
        curve_diff = np.linalg.norm(new_curve - self.curve)
        self.curve = new_curve
        return curve_diff


    def reparameterize(self):
        dist = np.cumsum(np.sqrt(np.sum(np.diff(self.curve, axis=0)**2, axis=1)))
        dist = np.insert(dist, 0, 0) / dist[-1] # normalize
        parameter = np.linspace(0, 1, self.n, endpoint=True)
        self.curve = interp1d(dist, self.curve, kind='cubic', axis=0)(parameter)


    def go(self, eps=0.01, n_max=300):
        curve_diff = self.update_curve()
        for i in range(n_max):
            curve_diff = self.update_curve()
            # Reparameterize the curve every 5 steps
            if i % 5 == 0:
                self.reparameterize()
            if curve_diff < eps:
                break


    @staticmethod
    def bilinear_interpolate(im, x, y):
        x = np.asarray(x)
        y = np.asarray(y)

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
