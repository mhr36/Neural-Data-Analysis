import numpy as np
from utils_for_max import Phi, Euler2fixedpt


def get_mu_sigma(W, W2, r, h, xi2, tau):
    # Find net input mean and variance given inputs
    mu = tau * (W @ r + h)
    sigma2 = tau * (W2 @ r + xi2)
    
    return mu, np.sqrt(sigma2)


def prob_func(P, w, theta):
    # Bernoulli parameter function
    return P * np.exp(- np.square(theta) / (2*w))


def theta_diff(a, b):
    d = (a - b) % 180
    q = d <= 90
    
    return d*q + (180-d)*(1-q)


class Model:
    def __init__(self, N=20, N_E=16):
        # Parameters chosen by us
        self.N = N
        self.N_E = N_E
        self.N_I = N - N_E

        # Parameters for 
        self.g_E = 10
        self.g_I = 10
        self.w_ff_E = 1
        self.w_ff_I = 1
        # Auxiliary time constants for excitatory and inhibitory
        T_alpha = 0.5
        self.T_E = 0.01
        self.T_I = 0.01 * T_alpha
        # Membrane time constants for excitatory and inhibitory
        tau_alpha = 1
        self.tau_E = 0.01
        self.tau_I = 0.01 * tau_alpha
        
        # Matrix of J, P, w coefficients for each weight in W
        self.J_full = np.zeros((N, N))
        self.P_full = np.zeros((N, N))
        self.w_full = np.zeros((N, N))
        
        self.preferred_orientations_full = np.zeros((N, N))
        
        self.orientations = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
        self.contrasts = [0., 0.0432773, 0.103411, 0.186966, 0.303066, 0.464386, 0.68854, 1.]
        
        self.update_cell_properties()
        
    def update_cell_properties():
        '''Set E-I type of the cells and preferred orientations'''
        # Set the cell types E or I
        cell_types = (np.concatenate([np.zeros(self.N_E, int), np.ones(self.N_I, int)]))
        self.cell_types = cell_types
        
        # Set the preferred orientations
        PO_E = np.linspace(0, 180, num=self.N_E, endpoint=False)
        PO_I = np.linspace(0, 180, num=self.N_I, endpoint=False)
        
        self.preferred_orientations = np.concatenate([PO_E, PO_I])
        
    def set_inputs(self, c, theta):  # --- NEEDS CHANGING ---
        '''Set the inputs based on the contrast and orientation of the stimulus'''
        
        g = np.concatenate([np.ones(self.N_E) * self.g_E,
                            np.ones(self.N_I) * self.g_I])
        
        w_ff = np.concatenate([np.ones(self.N_E) * self.w_ff_E,
                               np.ones(self.N_I) * self.w_ff_I])
        
        self.h = c * 20 * g * np.exp(-np.square(theta_diff(theta, self.preferred_orientations)) / (2 * np.square(w_ff)))
        self.xi2 = np.ones(self.N)
        
        self.r = np.zeros(self.N)
        
        return self.h, self.xi2
        
    def set_parameters(self, log_J, log_P, log_w):
        '''Set the main 3 parameter groups'''
        # Convert parameters to positive-only form
        J = np.exp(log_J) * [[1, -1],
                             [1, -1]]
        P = np.exp(log_P)
        w = np.exp(log_w)

        for i in range(self.N):
            for j in range(self.N):
                self.J_full[i, j] = J[cell_types[i], cell_types[j]]
                self.P_full[i, j] = P[cell_types[i], cell_types[j]]
                self.w_full[i, j] = w[cell_types[i], cell_types[j]]
                
                self.preferred_orientations_full[i, j] = self.preferred_orientations[i] - self.preferred_orientations[j]
                
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
        
    def generate_C_matrix(self):
        probabilities = prob_func(self.P_full, self.w_full, self.preferred_orientations_full)
        #print(probabilities)
        C = np.random.binomial(1, probabilities)
        np.fill_diagonal(C, 0)
        
        self.C = C
        
    def solve_fixed_point(self):
        
        def drdt_func(r):
            return self.T_inv * (Phi(*get_mu_sigma(self.W, self.W2, r, self.h, self.xi2, self.tau), self.tau) - r)
        
        self.r, did_converge = Euler2fixedpt(drdt_func, self.r)
        return did_converge
        
    def calculate_loss(self):
        '''Loss function from the paper'''
        pass
    
    def get_tuning_curves(self):
        '''With the current network, get tuning curves for all cells'''
        result = np.zeros([self.N, len(self.contrasts), len(self.orientations)])
        
        for i, c in enumerate(self.contrasts):
            for j, theta in enumerate(self.orientations):
                self.set_inputs(c, theta)
                
                if not self.solve_fixed_point:
                    raise Exception
                
                result[:, i, j] = self.r
        return result
        
