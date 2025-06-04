import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from model import SimpleINN2D
from utils import generate_toy_fk
from datetime import datetime

from analysis import plot_rigs, plot_uncertainty_histogram, scatter_pred_vs_true, scatter_fk_ik_validation

def compute_mmd(x, y, sigma=1.0):
    xx, yy, xy = x @ x.T, y @ y.T, x @ y.T
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    K_xx = torch.exp(- (rx.t() + rx - 2 * xx) / (2 * sigma ** 2))
    K_yy = torch.exp(- (ry.t() + ry - 2 * yy) / (2 * sigma ** 2))
    K_xy = torch.exp(- (rx.t() + ry - 2 * xy) / (2 * sigma ** 2))

    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd


def train():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    batch_size = 1024
    num_joints = 5
    rig_dim = 2 + num_joints
    input_dim = num_joints * 4

    num_epochs = 500000
    lambda_rig = 0.01
    lambda_recon = 1.0
    lambda_indep = 1.0
    lambda_restoration = 1e-5
    neutral_pose = torch.zeros((rig_dim,), dtype=torch.float32)


    results_dir = os.path.join(os.getcwd(), 'results', timestamp)
    os.makedirs(results_dir, exist_ok=True)

    model = SimpleINN2D(input_dim=input_dim, rig_dim=rig_dim)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of learnable parameters: {num_params}")

    optimizer = optim.Adam(model.parameters(), lr=4e-4)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        anchors, rigs = generate_toy_fk(batch_size, num_joints)
        joint_inputs = anchors.view(batch_size, -1)

        z = model(joint_inputs)
        recon = model(z, reverse=True)

        rigs_pred = model.predict_rig(z)
        latent_z = model.predict_latent(z)

        loss_rig = F.mse_loss(rigs_pred, rigs)
        loss_recon = F.mse_loss(recon, joint_inputs)
        loss_restoration = torch.mean(
            (rigs_pred - neutral_pose) ** 2 * latent_z.norm(dim=1, keepdim=True).detach()
        )


        # compute independence loss
        joint = torch.cat([rigs_pred, latent_z], dim=-1)
        target = torch.randn_like(joint)

        loss_indep = compute_mmd(joint, target)


        loss = lambda_rig * loss_rig + lambda_recon * loss_recon + lambda_indep * loss_indep + lambda_restoration * loss_restoration

        optimizer.zero_grad()
        # Grad norm clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            anchors_val, rigs_val = generate_toy_fk(batch_size, num_joints)
            joint_inputs_val = anchors_val.view(batch_size, -1)

            z_val = model(joint_inputs_val)
            recon_val = model(z_val, reverse=True)

            rigs_pred_val = model.predict_rig(z_val)
            latent_z_val = model.predict_latent(z_val)

            loss_rig_val = F.mse_loss(rigs_pred_val, rigs_val)
            loss_recon_val = F.mse_loss(recon_val, joint_inputs_val)
            loss_restoration_val = torch.mean((rigs_pred_val - neutral_pose) ** 2 * latent_z_val.norm(dim=1, keepdim=True))

            # compute independence loss
            joint_val = torch.cat([rigs_pred_val, latent_z_val], dim=-1)
            target_val = torch.randn_like(joint_val)

            loss_indep_val = compute_mmd(joint_val, target_val)

            val_loss = lambda_rig * loss_rig_val + lambda_recon * loss_recon_val + lambda_indep * loss_indep_val + lambda_restoration * loss_restoration_val

            val_losses.append(val_loss.item())

        if epoch % 10 == 0:
            # Print full training and validation loss components
            print(f"Epoch {epoch}:")
            # Before coefficients were applied
            print(f"  Train Loss: {loss.item():.4f} (Rig: {loss_rig.item():.4f}, Recon: {loss_recon.item():.4f}, "
                  f"Indep: {loss_indep.item():.4f}, Restoration: {loss_restoration.item():.4f})")
            print(f"  Val Loss:   {val_loss.item():.4f} (Rig: {loss_rig_val.item():.4f}, Recon: {loss_recon_val.item():.4f}, "
                  f"Indep: {loss_indep_val.item():.4f}, Restoration: {loss_restoration_val.item():.4f})")   
            # After coefficients were applied
            print(f"  Train Loss (weighted): {loss.item():.4f} (Rig: {lambda_rig * loss_rig.item():.4f}, Recon: {lambda_recon * loss_recon.item():.4f}, "
                  f"Indep: {lambda_indep * loss_indep.item():.4f}, Restoration: {lambda_restoration * loss_restoration.item():.4f})")
            print(f"  Val Loss (weighted):   {val_loss.item():.4f} (Rig: {lambda_rig * loss_rig_val.item():.4f}, Recon: {lambda_recon * loss_recon_val.item():.4f}, "
                  f"Indep: {lambda_indep * loss_indep_val.item():.4f}, Restoration: {lambda_restoration * loss_restoration_val.item():.4f})") 

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            # Save model state
            print(f"Saving model at epoch {epoch}...")
            torch.save(model.state_dict(), os.path.join(results_dir, f'model_epoch_{epoch:08d}.pth'))

            # Validation plots
            save_dir = os.path.join(results_dir, f'epoch_{epoch:08d}')
            os.makedirs(save_dir, exist_ok=True)

            model.eval()
            anchors_val, rigs_val = generate_toy_fk(64, num_joints)
            joint_inputs_val = anchors_val.view(64, -1)

            with torch.no_grad():
                latent_z = model(joint_inputs_val)
                pred_rigs = model.predict_rig(latent_z).cpu().numpy()

            plot_rigs(pred_rigs, rigs_val.numpy(), num_samples=5, save_dir=save_dir)
            plot_uncertainty_histogram(model.predict_latent(latent_z).cpu().numpy(), save_dir=save_dir)
            scatter_pred_vs_true(pred_rigs, rigs_val.numpy(), save_dir=save_dir)
            scatter_fk_ik_validation(
                true_rigs=rigs_val.numpy(),
                pred_rigs=pred_rigs,
                model=model,
                noise_std=0.05,
                neutral_pose=neutral_pose.numpy(),
                save_dir=save_dir
            )

            # Save losses plot
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Toy NIKI 2D Training Losses')
            plt.grid('both')
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'losses.png'))
            plt.close()

    return

if __name__ == "__main__":
    train()

