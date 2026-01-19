import numpy as np
import itertools

class ActionSpace:
    def __init__(self):
        self.M_options = [1, 3, 5] # Number of evidence items
        self.styles = ["concise", "detailed"] # Verbalization style
        self.weights = ["intent_focused", "distance_focused", "balanced"] # Path scoring preferences
        self.orders = ["score_desc", "diversity"] # Ordering strategy
        
        # Cartesian product: 3*2*3*2 = 36 Arms
        self.actions = [
            {"M": m, "style": s, "w": w, "order": o}
            for m in self.M_options 
            for s in self.styles 
            for w in self.weights 
            for o in self.orders
        ]
        self.n_arms = len(self.actions)

    def get_action(self, idx):
        return self.actions[idx]

class LinearThompsonSampling:
    """
    Standard Linear Thompson Sampling (Single Model) as per Eq 5-12.
    """
    def __init__(self, action_space, state_dim, lambda_reg=1.0, nu=1.0):
        self.action_space = action_space
        self.state_dim = state_dim
        # Calculate Z dimension: State (20) + Action Indicators (3+2+3+2 = 10) = 30
        self.z_dim = state_dim + 10 
        self.lambda_reg = lambda_reg
        self.nu = nu
        
        # Single A and b (Eq 7)
        self.A = lambda_reg * np.eye(self.z_dim)
        self.b = np.zeros(self.z_dim)
        self.A_inv = np.linalg.inv(self.A)
        
    def construct_z(self, state, action_idx):
        """
        Constructs z(s_t, a) = [ActionOneHots, NormalizedState]
        """
        # Get action dict
        act = self.action_space.get_action(action_idx)
        
        # One Hot Encoding of Action Parameters
        # M: [1, 3, 5] -> indices 0, 1, 2
        m_idx = self.action_space.M_options.index(act['M'])
        m_vec = np.zeros(3); m_vec[m_idx] = 1
        
        # Style: ['concise', 'detailed']
        s_idx = self.action_space.styles.index(act['style'])
        s_vec = np.zeros(2); s_vec[s_idx] = 1
        
        # W: ['intent', 'dist', 'bal']
        w_idx = self.action_space.weights.index(act['w'])
        w_vec = np.zeros(3); w_vec[w_idx] = 1
        
        # Order: ['score', 'div']
        o_idx = self.action_space.orders.index(act['order'])
        o_vec = np.zeros(2); o_vec[o_idx] = 1
        
        # Concat Action parts
        action_feat = np.concatenate([m_vec, s_vec, w_vec, o_vec])
        
        # Full z
        return np.concatenate([action_feat, state])

    def select_arm(self, state):
        """
        Eq 10: a_t = argmax theta_tilde^T z(s_t, a)
        """
        # 1. Posterior Mean (Eq 8)
        theta_hat = self.A_inv @ self.b
        
        # 2. Sample Theta (Eq 9)
        # Using SVD or Cholesky for stability if needed, but diag sample is faster approx?
        # Paper says Sample from Multivariate Normal.
        theta_tilde = np.random.multivariate_normal(theta_hat, (self.nu**2) * self.A_inv)
        
        # 3. Argmax over all actions
        best_arm = -1
        max_score = -float('inf')
        
        for idx in range(self.action_space.n_arms):
            z = self.construct_z(state, idx)
            score = theta_tilde.T @ z
            if score > max_score:
                max_score = score
                best_arm = idx
                
        return best_arm

    def update(self, arm_idx, state, reward):
        """
        Eq 11-12
        """
        z = self.construct_z(state, arm_idx)
        
        # A += z z^T
        self.A += np.outer(z, z)
        # b += r * z
        self.b += reward * z
        
        # Update inverse
        self.A_inv = np.linalg.inv(self.A)

    def save_params(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({'A': self.A, 'b': self.b}, f)
            
    def load_params(self, path):
        import pickle
        try:
            with open(path, 'rb') as f:
                params = pickle.load(f)
                self.A = params['A']
                self.b = params['b']
                self.A_inv = np.linalg.inv(self.A)
                print(f"Bandit parameters loaded from {path}")
        except FileNotFoundError:
            print(f"Warning: Bandit parameter file {path} not found. Starting fresh.")
