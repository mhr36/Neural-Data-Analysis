import numpy as np
import scipy.linalg as splinalg
from utils_for_max import Phi, Euler2fixedpt


def get_mu_sigma(W, W2, r, h, xi, tau):
    # Find net input mean and variance given inputs
    mu = tau * (W @ r) + h
    sigma2 = tau * (W2 @ r) + xi**2
    
    return mu, np.sqrt(sigma2)


def prob_func(P, w, theta):
    # Bernoulli parameter function
    return P * np.exp((np.cos(2*theta) - 1) / (2*w))  # [cos(2pi/L * x) -1] / [2pi/L * w] ^2


def theta_diff(a, b):
    # might delete this
    d = (a - b) % 180
    q = d <= 90
    
    return d*q + (180-d) * (1-q)


def block_matrix(V, d):
    # Return a matrix of blocks of values
    return np.block([[V[0,0]*np.ones((d[0], d[0])), V[0,1]*np.ones((d[0], d[1]))],
                     [V[1,0]*np.ones((d[1], d[0])), V[1,1]*np.ones((d[1], d[1]))]])


class Model:
    def __init__(self, N_E=16, N_I=4):  # 8000, 2000
        # Parameters chosen by us
        N = N_E + N_I
        self.N = N
        self.N_E = N_E
        self.N_I = N_I

        # Parameters for input stage
        self.g_E = 1
        self.g_I = 1
        self.w_ff_E = 1e-1
        self.w_ff_I = 1e-1
        self.sig_ext = 5
        
        # Auxiliary time constants for excitatory and inhibitory
        T_alpha = 0.5
        self.T_E = 0.01
        self.T_I = 0.01 * T_alpha
        
        # Membrane time constants for excitatory and inhibitory
        tau_alpha = 1
        self.tau_E = 0.01
        self.tau_I = 0.01 * tau_alpha        
        
        # Zero the state vector
        self.r = np.zeros(self.N)
        
        # Matrix of J, P, w coefficients for each weight in W
        self.J_full = np.zeros((N, N))
        self.P_full = np.zeros((N, N))
        self.w_full = np.zeros((N, N))
        
        # Matrix of differences in preferred orientations
        self.preferred_orientations_full = np.zeros((N, N))
        
        # Contrasts and orientations used in the experiment
        self.orientations = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
        self.contrasts = [0., 0.0432773, 0.103411, 0.186966, 0.303066, 0.464386, 0.68854, 1.]
        
        self.update_cell_properties()
        
    def update_cell_properties(self):
        '''Set E-I type of the cells and preferred orientations'''
        cell_types = (np.concatenate([np.zeros(self.N_E, int), np.ones(self.N_I, int)]))
        self.cell_types = cell_types
        
        # Set the preferred orientations, linearly spaced from 0-pi rad
        po_E = np.linspace(0, np.pi, num=self.N_E, endpoint=False)
        po_I = np.linspace(0, np.pi, num=self.N_I, endpoint=False)
        
        self.preferred_orientations = np.concatenate([po_E, po_I])
        
    def set_inputs(self, c, theta):
        '''Set the inputs based on the contrast and orientation of the stimulus'''
        
        # Distribute parameters over all neurons based on type
        g = np.concatenate([np.ones(self.N_E) * self.g_E,
                            np.ones(self.N_I) * self.g_I])
        
        w_ff = np.concatenate([np.ones(self.N_E) * self.w_ff_E,
                               np.ones(self.N_I) * self.w_ff_I])
        
        self.h = c * 20 * g * np.exp((np.cos(2*(theta-self.preferred_orientations)) - 1) / (2 * np.square(w_ff)))
        self.xi = np.ones(self.N) * self.sig_ext
        
        return self.h, self.xi
        
    def set_parameters(self, log_J, log_P, log_w):
        '''Set the main 3 parameter groups'''
        # Convert parameters to positive-only form
        J = np.exp(log_J) * [[1, -1],
                             [1, -1]]
        P = np.exp(log_P)
        w = np.exp(log_w)
        
        # Create matrices with blocks of parameters
        self.J_full = block_matrix(J, [self.N_E, self.N_I])
        self.P_full = block_matrix(P, [self.N_E, self.N_I])
        self.w_full = block_matrix(w, [self.N_E, self.N_I])
        
        # Create matrix of differences between preferred orientations
        po_EE = splinalg.toeplitz(self.preferred_orientations[:self.N_E])
        po_II = splinalg.toeplitz(self.preferred_orientations[self.N_E:])
        
        po_EI = np.absolute(np.subtract.outer(
            self.preferred_orientations[:self.N_E],
            self.preferred_orientations[self.N_E:]
        ))
        
        self.preferred_orientations_full = np.block([[po_EE, po_EI],
                                                     [po_EI.T, po_II]])
        
    def generate_C_matrix(self):
        probabilities = prob_func(self.P_full, self.w_full, self.preferred_orientations_full)
        C = np.random.binomial(1, probabilities)  # This needs to be changed to a continuous form
        np.fill_diagonal(C, 0)
        
        self.C = C
                
    def generate_network(self):
        '''Randomly generate network'''
        self.generate_C_matrix()

        # Auxiliary time constant vector for all cells
        self.T = self.cell_types * self.T_I + (1-self.cell_types) * self.T_E
        self.T_inv = np.reciprocal(self.T)

        # Membrane time constant vector for all cells
        self.tau = self.cell_types * self.tau_I + (1-self.cell_types) * self.tau_E

        # Weight matrix and squared weight matrix
        self.W = self.J_full * self.C
        self.W2 = np.square(self.W)
        
    def solve_fixed_point(self):
        
        # Define the function to be solved for
        def drdt_func(r):
            return self.T_inv * (Phi(*get_mu_sigma(self.W, self.W2, r, self.h, self.xi, self.tau), self.tau) - r)
        
        # Solve using Euler
        self.r, did_converge = Euler2fixedpt(drdt_func, self.r)
        return did_converge
    
    def r_change(self):
        # DELETE THIS
        return Phi(*get_mu_sigma(self.W, self.W2, self.r, self.h, self.xi, self.tau), self.tau)
        
    def calculate_loss(self):
        '''Loss function from the paper'''
        pass
    
    def get_tuning_curves(self):
        '''With the current network, get tuning curves for all cells'''
        result = np.zeros([self.N, len(self.contrasts), len(self.orientations)])
        
        # Iterate through all contrasts and orientations
        for i, c in enumerate(self.contrasts):
            for j, theta in enumerate(self.orientations):
                # Set up the model
                self.set_inputs(c, theta)
                # Find fixed point
                if not self.solve_fixed_point():
                    raise Exception
                
                result[:, i, j] = self.r
        return result
        
