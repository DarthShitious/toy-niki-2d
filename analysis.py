# analysis.py
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def plot_rigs(pred_rigs, true_rigs, num_samples=5, save_dir=None):
    fig, axs = plt.subplots(num_samples, 1, figsize=(8, num_samples * 2))

    for i in range(num_samples):
        axs[i].plot(true_rigs[i], label='True Rig', marker='o')
        axs[i].plot(pred_rigs[i], label='Predicted Rig', marker='x')
        axs[i].set_title(f'Sample {i}')
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()

    if save_dir is None:
        plt.show()
    else:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'rigs_plot.png'))
        plt.close()

def plot_uncertainty_histogram(latent_z, save_dir=None):
    uncertainty = np.linalg.norm(latent_z, axis=-1)

    plt.figure(figsize=(8, 4))
    plt.hist(uncertainty, bins=30, alpha=0.7, color='purple')
    plt.xlabel('Uncertainty (Norm of Latent Error)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Uncertainty')
    plt.grid(True)

    if save_dir is None:
        plt.show()
    else:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'uncertainty_histogram.png'))
        plt.close()

def scatter_pred_vs_true(pred_rigs, true_rigs, save_dir=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(true_rigs.flatten(), pred_rigs.flatten(), alpha=0.5)
    plt.xlabel('True Rig Values')
    plt.ylabel('Predicted Rig Values')
    plt.title('Scatter Plot: Predicted vs True')
    plt.grid(True)
    plt.plot([-1, 1], [-1, 1], color='red', linestyle='--')

    if save_dir is None:
        plt.show()
    else:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'scatter_pred_vs_true.png'))
        plt.close()

def forward_kinematics(rig):
    num_joints = rig.shape[-1] - 2
    lengths = np.ones(num_joints) * 1.0

    x, y = rig[0], rig[1]
    total_angle = 0.0
    xs = [x]
    ys = [y]

    for j in range(num_joints):
        total_angle += rig[2 + j]
        x += lengths[j] * np.cos(total_angle)
        y += lengths[j] * np.sin(total_angle)
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

def scatter_fk_ik_validation(true_rigs, pred_rigs, model, noise_std=0.01, neutral_pose=None, save_dir=None):
    batch_size = true_rigs.shape[0]

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for i in range(min(batch_size, 5)):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Clean reconstruction
        true_fk_xs, true_fk_ys = forward_kinematics(true_rigs[i])
        pred_fk_xs, pred_fk_ys = forward_kinematics(pred_rigs[i])

        # Inject noise
        true_anchor = rig_to_anchor(true_rigs[i])
        noisy_anchor = true_anchor + np.random.normal(0, noise_std, true_anchor.shape)

        noisy_anchor_tensor = torch.tensor(noisy_anchor, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            z_noisy = model(noisy_anchor_tensor)
            recon_rig_noisy = model.predict_rig(z_noisy).squeeze(0).cpu().numpy()
        noisy_fk_xs, noisy_fk_ys = forward_kinematics(recon_rig_noisy)

        # Neutral pose FK
        if neutral_pose is not None:
            neutral_fk_xs, neutral_fk_ys = forward_kinematics(neutral_pose)

        # Plot clean FK and IK
        axs[0].plot(true_fk_xs, true_fk_ys, label='True FK', marker='o')
        axs[0].plot(pred_fk_xs, pred_fk_ys, label='Predicted IK', marker='x')
        if neutral_pose is not None:
            axs[0].plot(neutral_fk_xs, neutral_fk_ys, label='Neutral Pose', linestyle='--')
        axs[0].set_title('Clean Anchor Reconstruction')
        axs[0].set_aspect('equal')
        axs[0].grid(True)
        axs[0].legend()

        # Plot noisy FK and IK
        axs[1].plot(true_fk_xs, true_fk_ys, label='True FK', marker='o')
        axs[1].plot(noisy_fk_xs, noisy_fk_ys, label='Noisy IK', marker='x')
        if neutral_pose is not None:
            axs[1].plot(neutral_fk_xs, neutral_fk_ys, label='Neutral Pose', linestyle='--')
        axs[1].set_title('Noisy Anchor Reconstruction')
        axs[1].set_aspect('equal')
        axs[1].grid(True)
        axs[1].legend()

        plt.tight_layout()

        if save_dir is None:
            plt.show()
        else:
            filename = os.path.join(save_dir, f'validation_{i}.png')
            plt.savefig(filename)
            plt.close()

def rig_to_anchor(rig):
    num_joints = rig.shape[0] - 2
    lengths = np.ones(num_joints) * 1.0

    x, y = rig[0], rig[1]
    total_angle = 0.0
    anchor_list = []

    for j in range(num_joints):
        total_angle += rig[2 + j]
        x += lengths[j] * np.cos(total_angle)
        y += lengths[j] * np.sin(total_angle)
        anchor_list.append([x, y, np.cos(total_angle), np.sin(total_angle)])

    return np.array(anchor_list).flatten()