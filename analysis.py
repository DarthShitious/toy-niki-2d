# analysis.py
import matplotlib.pyplot as plt
import numpy as np

def plot_rigs(pred_rigs, true_rigs, num_samples=5):
    """
    Plot a few predicted vs true rig vectors for inspection.
    """
    fig, axs = plt.subplots(num_samples, 1, figsize=(8, num_samples * 2))
    
    for i in range(num_samples):
        axs[i].plot(true_rigs[i], label='True Rig', marker='o')
        axs[i].plot(pred_rigs[i], label='Predicted Rig', marker='x')
        axs[i].set_title(f'Sample {i}')
        axs[i].legend()
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_uncertainty_histogram(latent_z):
    """
    Plot a histogram of uncertainty magnitudes (norm of latent error).
    """
    uncertainty = np.linalg.norm(latent_z, axis=-1)

    plt.figure(figsize=(8, 4))
    plt.hist(uncertainty, bins=30, alpha=0.7, color='purple')
    plt.xlabel('Uncertainty (Norm of Latent Error)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Uncertainty')
    plt.grid(True)
    plt.show()

def scatter_pred_vs_true(pred_rigs, true_rigs):
    """
    Scatter plot of predicted vs true values.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(true_rigs.flatten(), pred_rigs.flatten(), alpha=0.5)
    plt.xlabel('True Rig Values')
    plt.ylabel('Predicted Rig Values')
    plt.title('Scatter Plot: Predicted vs True')
    plt.grid(True)
    plt.plot([-1, 1], [-1, 1], color='red', linestyle='--')  # perfect prediction line
    plt.show()
