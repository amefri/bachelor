import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def generate_sample(size, mean=0, std=1):
    return np.random.normal(mean, std, size)

def weighted_sample(size, weight_func, mean=0, std=1):
    x = np.random.normal(mean, std, size * 2)
    w = weight_func(x)
    return x[np.random.random(size * 2) < w / np.max(w)][:size]

def polynomial_weight(x, a):
    return 1 + a * x**2

def exponential_weight(x, a):
    return np.exp(a * x)

def calculate_kl_divergence(p, q):
    # Use kernel density estimation for smoother pdf estimates
    kde_p = stats.gaussian_kde(p)
    kde_q = stats.gaussian_kde(q)
    x = np.linspace(min(p.min(), q.min()), max(p.max(), q.max()), 1000)
    return np.sum(kde_p(x) * (np.log(kde_p(x)) - np.log(kde_q(x)))) * (x[1] - x[0])

def calculate_kld_curve(sample_sizes, reference_dist, test_dist, num_runs=10):
    kld_values = []
    for size in sample_sizes:
        klds = []
        for _ in range(num_runs):
            p = reference_dist(size)
            q = test_dist(size)
            klds.append(calculate_kl_divergence(p, q))
        kld_values.append(np.mean(klds))
    return kld_values

def plot_kld_curves(sample_sizes, curves, labels):
    plt.figure(figsize=(12, 7))
    for curve, label in zip(curves, labels):
        plt.plot(sample_sizes, curve, label=label, marker='o', markersize=4)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sample Size')
    plt.ylabel('KL Divergence')
    plt.legend()
    plt.title('KL Divergence vs Sample Size')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.show()

# Main execution
n = 10000
sample_sizes = np.logspace(1, 5, 30).astype(int)

# Define distribution generators
unweighted_dist = lambda size: generate_sample(size)
poly_weighted_dist = lambda size: weighted_sample(size, lambda x: polynomial_weight(x, 0.5))
exp_weighted_dist = lambda size: weighted_sample(size, lambda x: exponential_weight(x, 0.2))

# Calculate KLD curves
unweighted_curve = calculate_kld_curve(sample_sizes, unweighted_dist, unweighted_dist)
poly_weighted_curve = calculate_kld_curve(sample_sizes, unweighted_dist, poly_weighted_dist)
exp_weighted_curve = calculate_kld_curve(sample_sizes, unweighted_dist, exp_weighted_dist)

# Plot results
plot_kld_curves(sample_sizes, 
                [unweighted_curve, poly_weighted_curve, exp_weighted_curve],
                ['Unweighted', 'Polynomial Weighted', 'Exponential Weighted'])