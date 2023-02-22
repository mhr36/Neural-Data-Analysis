#import numpy as np
import jax.numpy as np
import jax.random as jrand
import scipy.linalg as splinalg
from utils_for_max import Phi, Euler2fixedpt


prng = jrand.PRNGKey(1)


def circ_gauss(x, w):
    # Circular Gaussian from 0 to 180 deg
    return np.exp((np.cos(x * np.pi/90) - 1) / np.square(np.pi/90 * w))  # [cos(2pi/L * x) -1] / [2pi/L * w] ^2


def kernel(diff_square, w=1):
    '''Problem: This operation is now comparing two tuning curves instead of two data points??'''
    x = diff_square.mean(axis=[2,3])
    return np.exp(x / np.square(np.pi/90 * w))


def get_mu_sigma(W, W2, r, h, xi, tau):
    # Find net input mean and variance given inputs
    mu = tau * (W @ r) + h
    sigma2 = tau * (W2 @ r) + xi**2
    
    return mu, np.sqrt(sigma2)


def MMD(X, Y):
    # Maximum Mean Discrepancy
    N = len(X)
    M = len(Y)
    
    # DS: Difference Squared
    DS_XX = np.square(X[None, :, :, :] - X[:, None, : , :])
    DS_XY = np.square(X[None, :, :, :] - Y[:, None, : , :])
    DS_YY = np.square(Y[None, :, :, :] - Y[:, None, : , :])
    
    sumXX = np.sum(kernel(DS_XX))
    sumXY = np.sum(kernel(DS_XY))
    sumYY = np.sum(kernel(DS_YY))
    
    return sumXX/(N*N) - 2*sumXY/(N*M) + sumYY/(M*M)


def prob_func(P, w, theta):
    # Bernoulli parameter function
    return P * circ_gauss(theta, w)


def random_matrix(probabilities):
    # Produce continuous Bernoulli substitute
    rmat = jrand.uniform(prng, probabilities.shape)
    return 1 / (1 + np.exp(32*(rmat-probabilities)))  # Factor 32 can change


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
        
        self.lamb =1

        # Parameters for input stage
        self.g_E = 1
        self.g_I = 1
        self.w_ff_E = 30
        self.w_ff_I = 30
        self.sig_ext = 5
        
        # Auxiliary time constants for excitatory and inhibitory
        T_alpha = 0.5
        self.T_E = 0.01
        self.T_I = 0.01 * T_alpha
        
        # Membrane time constants for excitatory and inhibitory
        tau_alpha = 1
        self.tau_E = 0.01
        self.tau_I = 0.01 * tau_alpha
        
        #Refractory periods for exitatory and inhibitory
        self.tau_ref_E = 0.005
        self.tau_ref_I = 0.001
        self.tau_ref = np.concatenate([
            self.tau_ref_E * np.ones(self.N_E),
            self.tau_ref_I * np.ones(self.N_I)])
        
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
        po_E = np.linspace(0, 180, num=self.N_E, endpoint=False)
        po_I = np.linspace(0, 180, num=self.N_I, endpoint=False)
        
        self.preferred_orientations = np.concatenate([po_E, po_I])
        
    def set_inputs(self, c, theta):
        '''Set the inputs based on the contrast and orientation of the stimulus'''
        
        # Distribute parameters over all neurons based on type
        g = np.concatenate([np.ones(self.N_E) * self.g_E,
                            np.ones(self.N_I) * self.g_I])
        
        w_ff = np.concatenate([np.ones(self.N_E) * self.w_ff_E,
                               np.ones(self.N_I) * self.w_ff_I])
        
        self.h = c * 20 * g * circ_gauss(theta-self.preferred_orientations, w_ff)
        self.xi = np.ones(self.N) * self.sig_ext
        
        return self.h, self.xi
        
    def set_parameters(self, log_J, log_P, log_w):
        '''Set the main 3 parameter groups'''
        # Convert parameters to positive-only form
        J = np.exp(log_J) * np.array([[1, -1],
                                     [1, -1]])
        P = np.exp(log_P)
        w = np.exp(log_w)
        
        # Create matrices with blocks of parameters
        self.J_full = block_matrix(J, [self.N_E, self.N_I])
        self.P_full = block_matrix(P, [self.N_E, self.N_I])
        self.w_full = block_matrix(w, [self.N_E, self.N_I])
        
        # Create matrix of differences between preferred orientations
        po_EE = splinalg.toeplitz(self.preferred_orientations[:self.N_E])
        po_II = splinalg.toeplitz(self.preferred_orientations[self.N_E:])
        
        po_EI = self.preferred_orientations[:self.N_E, None] - self.preferred_orientations[None, self.N_E:]
        
        self.preferred_orientations_full = np.block([[po_EE, po_EI],
                                                     [po_EI.T, po_II]])
        
    def generate_C_matrix(self):
        probabilities = prob_func(self.P_full, self.w_full, self.preferred_orientations_full)
        C = random_matrix(probabilities) * (1-np.eye(self.N))
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
            return self.T_inv * (Phi(*get_mu_sigma(self.W, self.W2, r, self.h, self.xi, self.tau), self.tau, tau_ref=self.tau_ref) - r)
        
        # Solve using Euler
        self.r, avg_step = Euler2fixedpt(drdt_func, self.r)
        return avg_step
    
    def r_change(self):
        # DELETE THIS
        return Phi(*get_mu_sigma(self.W, self.W2, self.r, self.h, self.xi, self.tau), self.tau)
        
    def calculate_loss(self, data):
        '''Loss function from the paper'''
        # REPLACE FOLLOWING SUBSAMPLING WITH jrand.choice(A, n)
        loss = MMD(self.tuning_curves[0::100], data[0::100]) + self.lamb * (np.maximum(1, self.avg_step) - 1)
        
        return loss
    
    def get_tuning_curves(self):
        '''With the current network, get tuning curves for all cells'''
        result = np.zeros([self.N, len(self.contrasts), len(self.orientations)])
        
        avg_step_sum = 0
        
        # Iterate through all contrasts and orientations
        for i, c in enumerate(self.contrasts):
            print('contrasts: '+str(i)+'/'+str(len(self.contrasts)))
            for j, theta in enumerate(self.orientations):
                # Set up the model
                self.set_inputs(c, theta)
                # Find fixed point
                avg_step_sum += self.solve_fixed_point()
                
                
                
                result.at[:, i, j].set(self.r)
                
        self.avg_step = avg_step_sum / (len(self.contrasts) * len(self.orientations))
        self.tuning_curves = result
        return result
        
