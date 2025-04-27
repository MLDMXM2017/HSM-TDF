# @inproceedings{HSM-TDF,  
#   title={Tongue Diagnosis Framework for Fatty Liver Disease Severity Classification Using Kolmogorov-Arnold Network},  
#   link={https://github.com/MLDMXM2017/HSM-TDF}  
# }  

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma, betaln
from scipy.stats import beta
from sklearn.base import BaseEstimator, DensityMixin
from sklearn.utils.validation import check_array, check_is_fitted

class BetaMixture(BaseEstimator, DensityMixin):
   
    def __init__(self, n_components=1, tol=1e-3, max_iter=100, random_state=None):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

    def _beta_pdf(self, X, alpha, beta):
        """Compute the Beta distribution PDF for given alpha and beta."""
        X = np.clip(X, 1e-8, 0.9999999)
        log_pdf = (alpha - 1) * np.log(X) + (beta - 1) * np.log(1 - X) - betaln(alpha, beta)
        return np.exp(log_pdf)        

    def _initialize_parameters(self, X):
        """Initialize mixture weights, alpha, and beta parameters."""
        n_samples = X.shape[0]
        # rng = np.random.RandomState(self.random_state)
        self.weights_ = np.ones(self.n_components) / self.n_components
        self.alphas_ = np.array([1.0001, 2.5])
        self.betas_ = np.array([2.5, 1.0001])
        if np.any(X < 0) or np.any(X > 1):
            raise ValueError("Data must be in the range [0, 1] for Beta distribution.")
    
    def _e_step(self, X):
        """E-step: Compute responsibilities (posterior probabilities)."""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights_[k] * self._beta_pdf(X, self.alphas_[k], self.betas_[k]).reshape(-1)
        
        responsibilities += 1e-8  
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities
    
    def _m_step_iter(self, X, resp):
        Nk = np.sum(resp, axis=0)
        self.weights_ = Nk / np.sum(Nk)

        for k in range(self.n_components):
            rk = resp[:, k]
            Nk_k = Nk[k]
            x_mean = np.sum(rk * X) / Nk_k
            x_var = np.sum(rk * (X - x_mean) ** 2) / Nk_k
            common_term = x_mean * (1 - x_mean) / (x_var + 1e-6) - 1
            alpha = max(x_mean * common_term, 1.01)
            beta_ = max((1 - x_mean) * common_term, 1.01)
            self.alphas_[k] = alpha
            self.betas_[k] = beta_

    def _m_step_grad(self, X, responsibilities):
        """M-step: Update mixture weights, alpha, and beta."""
        n_samples = X.shape[0]
        
        Nk = responsibilities.sum(axis=0) + 1e-10 
        self.weights_ = Nk / n_samples
        
        for k in range(self.n_components):
            resp_k = responsibilities[:, k: k+1]
            Nk_k = Nk[k]
            
            log_x = np.log(X + 1e-10) 
            log_1_x = np.log(1 - X + 1e-10)
            
            alpha = self.alphas_[k]
            beta = self.betas_[k]
            
            psi_alpha = digamma(alpha)
            psi_beta = digamma(beta)
            psi_alpha_beta = digamma(alpha + beta)
            
            g_alpha = Nk_k * (psi_alpha_beta - psi_alpha + np.average(log_x, weights=resp_k))
            g_beta = Nk_k * (psi_alpha_beta - psi_beta + np.average(log_1_x, weights=resp_k))
            
            alpha += g_alpha / (Nk_k * (digamma(alpha + beta) - digamma(alpha)))
            beta += g_beta / (Nk_k * (digamma(alpha + beta) - digamma(beta)))
            
            alpha = max(alpha, 1.01)
            beta = max(beta, 1.01)
            
            self.alphas_[k] = alpha
            self.betas_[k] = beta

    def _compute_log_likelihood(self, X):
        """Compute the log-likelihood of the data."""
        n_samples = X.shape[0]
        likelihood = np.zeros((n_samples,1))
        
        for k in range(self.n_components):
            likelihood += self.weights_[k] * self._beta_pdf(X, self.alphas_[k], self.betas_[k])
        
        return np.sum(np.log(likelihood + 1e-10))

    def fit_iter(self, X):
        X = np.asarray(X).ravel()
        X = np.clip(X, 1e-8, 1 - 1e-8)
        self._initialize_parameters(X)
        prev_log_likelihood = None

        for iteration in range(self.max_iter):
            resp = self._e_step(X)
            self._m_step_iter(X, resp)

            # Compute log-likelihood
            ll = np.sum(np.log(np.sum([
                self.weights_[k] * self._beta_pdf(X, self.alphas_[k], self.betas_[k])
                for k in range(self.n_components)
            ], axis=0)))

            if prev_log_likelihood is not None and abs(ll - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = ll
        return self
    
    def fit_grad(self, X):
        X = check_array(X, ensure_2d=False)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        
        self._initialize_parameters(X)
        
        log_likelihood_old = -np.inf
        for iteration in range(self.max_iter):
            # E-step
            responsibilities = self._e_step(X)
            
            # M-step
            self._m_step_grad(X, responsibilities)
            
            log_likelihood = self._compute_log_likelihood(X)
            if abs(log_likelihood - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood
        
        self.converged_ = iteration < self.max_iter - 1
        return self    

    def fit(self, X, y=None):
        if X.shape[0] > 1666:
            return self.fit_grad(X)
        else:
            return self.fit_iter(X)

    def expected_val(self):
        return self.alphas_ / (self.alphas_ + self.betas_)

    def mode(self):
        return (self.alphas_ - 1)/(self.alphas_ + self.betas_ - 2)

    def predict_hard(self, X):
        X = np.clip(X, 1e-8, 0.9999999).reshape(-1, 1)
        mode = self.expected_val() # .mode()
        minus = X - mode
        sign = minus[:, 0] * minus[:, 1]
        return np.where(sign < 0, 1, 0) 

    def predict_easy(self, X):
        X = np.clip(X, 1e-8, 0.9999999).reshape(-1, 1)
        mode = self.expected_val()
        minus = X.reshape(-1) - mode.max()
        return np.where(minus > 0, 1, 0)

    def draw_hist_beta_prob(self, input_sim, labels, name, c_pointed=-1):
        
        colorbar = ['blue', 'orange', 'red']
        class_name = ['none', 'mild', 'severe']
        bins = list(np.arange(-0.01, 1.01, 0.02))
        plt.figure(figsize=(4, 4))
        y_i = input_sim.reshape(-1)
        plt.hist(y_i, bins=bins, density=True, alpha=0.66, histtype='step', color=colorbar[c_pointed], label=class_name[c_pointed])
        plt.hist(y_i, bins=bins, density=True, alpha=0.33, color=colorbar[c_pointed])
        mode = self.expected_val()    
        
        x = np.linspace(0, 1, 1000).reshape(-1, 1)
        pdf_sum = 0
        for i in range(2):
            pdf = self.weights_[i] * beta.pdf(x, self.alphas_[i], self.betas_[i])
            pdf_sum += pdf
            plt.plot(x, pdf, linestyle='--', label=f'Component {i+1}')
            plt.axvline(x=mode[i], color='green', linestyle='-.', label=f'Mode {i+1}: {mode[i]:.2f}')
        plt.plot(x, pdf_sum, label='Sum of Components')

        plt.xlabel('LLC')
        plt.ylabel('Sample Density')
        plt.xlim(-0.01, 1.01)
        plt.title('Distribution of LLC and Beta')
        plt.legend()
        save_dir = './figure/'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + name + '.svg')
        plt.close()

    def predict_component_proba(self, X):
        n_samples = X.shape[0]
        X = np.clip(X, 1e-8, 0.9999999).reshape(-1, 1)
        likelihood = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            likelihood[:, k:k+1] = self.weights_[k] * self._beta_pdf(X, self.alphas_[k], self.betas_[k])
        
        return likelihood / likelihood.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Predict the component each sample belongs to."""
        check_is_fitted(self, ['weights_', 'alphas_', 'betas_'])
        X = check_array(X, ensure_2d=False)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def score(self, X, y=None):
        """Compute the per-sample average log-likelihood."""
        check_is_fitted(self, ['weights_', 'alphas_', 'betas_'])
        X = check_array(X, ensure_2d=False)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        
        return self._compute_log_likelihood(X) / X.shape[0]
