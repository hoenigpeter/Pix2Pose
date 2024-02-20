import torch
import torch.nn as nn
import numpy as np

class TransformerLoss(nn.Module):
    def __init__(self, sym=0):
        super(TransformerLoss, self).__init__()
        self.sym = sym

    def forward(self, x):
        y_pred = x[0]
        y_recont_gt = x[1]
        y_prob_pred = torch.squeeze(x[2], dim=3)
        y_prob_gt = x[3]
        visible = (y_prob_gt > 0.5).type_as(y_pred)
        #print("visible: ", visible.shape)
        visible = torch.squeeze(visible, dim=1)
        #print("visible: ", visible.shape)
        # Generate transformed values using sym
        if len(self.sym) > 1:
            loss_sums = torch.zeros(1).type_as(y_pred)
            loss_xyzs = torch.zeros(1).type_as(y_pred)

            for sym_id, transform in enumerate(self.sym):
                tf_mat = torch.tensor(transform, dtype=y_recont_gt.dtype)
                y_gt_transformed = torch.transpose(torch.matmul(tf_mat, torch.transpose(torch.reshape(y_recont_gt, [-1, 3]), 0, 1)), 0, 1)
                y_gt_transformed = torch.reshape(y_gt_transformed, [-1, 128, 128, 3])
                loss_xyz_temp = torch.sum(torch.abs(y_gt_transformed - y_pred), dim=3) / 3
                loss_sum = torch.sum(loss_xyz_temp, dim=[1, 2])

                if sym_id > 0:
                    loss_sums = torch.cat([loss_sums, torch.unsqueeze(loss_sum, 0)], dim=0)
                    loss_xyzs = torch.cat([loss_xyzs, torch.unsqueeze(loss_xyz_temp, 0)], dim=0)
                else:
                    loss_sums = torch.unsqueeze(loss_sum, 0)
                    loss_xyzs = torch.unsqueeze(loss_xyz_temp, 0)

            min_values, _ = torch.min(loss_sums, dim=0, keepdim=True)
            loss_switch = (loss_sums == min_values).type_as(y_pred)
            loss_xyz = torch.unsqueeze(torch.unsqueeze(loss_switch, 2), 3) * loss_xyzs
            loss_xyz = torch.sum(loss_xyz, dim=0)
        else:
            loss_xyz = torch.sum(torch.abs(y_recont_gt - y_pred), dim=1) / 3

        loss_xyz = loss_xyz.unsqueeze(1)
        #print("loss_xyz: ", loss_xyz.shape)
        prob_loss = torch.square(y_prob_pred - torch.min(loss_xyz, dim=1).values)
        loss_invisible = (1 - visible) * loss_xyz
        loss_visible = visible * loss_xyz
        loss = loss_visible * 3 + loss_invisible + 0.5 * prob_loss
        #print("loss shape: ", loss.shape)
        loss = torch.mean(torch.mean(loss, dim=[2, 3]))
        return loss

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class DiffusionModel(nn.Module):
    def __init__(self, num_steps, in_channels, out_channels, noise_std):
        super(DiffusionModel, self).__init__()
        self.num_steps = num_steps
        self.noise_std = noise_std
        
        # Encoder blocks
        self.encoder = nn.ModuleList([
            EncoderBlock(in_channels, out_channels),
            EncoderBlock(out_channels, out_channels),
            EncoderBlock(2 * out_channels, 2 * out_channels),
            EncoderBlock(2 * out_channels, 2 * out_channels),
            EncoderBlock(2 * 2 * out_channels, 2 * 2 * out_channels),
            EncoderBlock(2 * 2 * out_channels, 2 * 2 * out_channels),
            EncoderBlock(2 * 2 * out_channels, 2 * 2 * out_channels),
            EncoderBlock(2 * 2 * out_channels, 2 * 2 * out_channels)
        ])

        # Diffusion steps
        self.diffusion_steps = nn.ModuleList([
            DiffusionStep(2 * 2 * out_channels, 2 * 2 * out_channels, kernel_size=5, stride=2, padding=2, noise_std=noise_std)
            for _ in range(num_steps)
        ])

        # Final decoder
        self.final_decoder = nn.Sequential(
            nn.ConvTranspose2d(2 * 2 * out_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

        # Pixel probability branch
        self.pixel_prob_branch = nn.Sequential(
            nn.ConvTranspose2d(2 * 2 * out_channels, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoding
        for encoder_block in self.encoder:
            x = encoder_block(x)
        
        # Diffusion steps
        for step in range(self.num_steps):
            x = self.diffusion_steps[step](x)
        
        # Final decoding
        decoded = self.final_decoder(x)
        pixel_prob = self.pixel_prob_branch(x)

        return decoded, pixel_prob


class DiffusionStep(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, noise_std):
        super(DiffusionStep, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        self.noise_std = noise_std

    def forward(self, x):
        # Add noise
        noise = torch.randn_like(x) * self.noise_std
        x = x + noise

        # Apply diffusion step
        x = self.relu(self.bn(self.conv(x)))
        return x
    
