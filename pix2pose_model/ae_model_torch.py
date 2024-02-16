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
        print("visible: ", visible.shape)
        visible = torch.squeeze(visible, dim=1)
        print("visible: ", visible.shape)
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
            loss_xyz = torch.sum(torch.abs(y_recont_gt - y_pred), dim=3) / 3
        print(loss_xyz.shape)
        print(y_prob_pred.shape)
        print(torch.min(loss_xyz, dim=1).values.shape)
        prob_loss = torch.square(y_prob_pred - torch.min(loss_xyz, dim=1).values)
        loss_xyz = loss_xyz.unsqueeze(2)
        loss_invisible = (1 - visible) * loss_xyz
        loss_visible = visible * loss_xyz
        loss = loss_visible * 3 + loss_invisible + 0.5 * prob_loss
        loss = torch.mean(loss, dim=[1, 2])
        return loss

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding):
        super(DecoderBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=2, output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))

class AE_Model(nn.Module):
    def __init__(self):
        super(AE_Model, self).__init__()
        self.encoder = nn.ModuleList([
            EncoderBlock(3, 64),
            EncoderBlock(3, 64),            
            EncoderBlock(2*64, 128),
            EncoderBlock(2*64, 128),
            EncoderBlock(2*128, 128),
            EncoderBlock(2*128, 128),
            EncoderBlock(2*128, 256),
            EncoderBlock(2*128, 256)
        ])

        self.bottleneck = nn.Linear(8 * 8 * 512, 256)

        self.decoder = nn.ModuleList([
            nn.Linear(256, 8 * 8 * 256),
            nn.Unflatten(1, (256, 8, 8)),
            DecoderBlock(256, 128, 5, 2, 1),   # fuer d1
            DecoderBlock(256, 256, 5, 1, 0),   # fuer d1_uni
            DecoderBlock(256, 128, 5, 2, 1),   # fuer d2
            DecoderBlock(256, 128, 5, 1, 0),   # fuer d2_uni           
            DecoderBlock(128, 64, 5, 2, 1),   # fuer d3
            DecoderBlock(128, 128, 5, 1, 0),   # fuer d3_uni   
        ])

        self.final_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

        self.pixel_prob_branch = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        f1_1 = self.encoder[0](x)
        f1_2 = self.encoder[1](x)
        f1 = torch.cat([f1_1, f1_2], dim=1)
        #print("f1:", f1.shape)
        
        f2_1 = self.encoder[2](f1)
        f2_2 = self.encoder[3](f1)
        f2 = torch.cat([f2_1, f2_2], dim=1)
        #print("f2:", f2.shape)
        
        f3_1 = self.encoder[4](f2)
        f3_2 = self.encoder[5](f2)
        f3 = torch.cat([f3_1, f3_2], dim=1)
        #print("f3:", f3.shape)
        
        f4_1 = self.encoder[6](f3)
        f4_2 = self.encoder[7](f3)
        f4 = torch.cat([f4_1, f4_2], dim=1)
        #print("f3:", f3.shape)

        x = f4.view(f4.size(0), -1)
        x = self.bottleneck(x)
        #print("bottleneck:", x.shape)
        
        #x = x.view(x.size(0), 256, 8, 8)
        d1 = self.decoder[0](x)
        #print("d1:", d1.shape)
        d1 = self.decoder[1](d1)
        #print("d1:", d1.shape)        
        d1 = self.decoder[2](d1)        
        #print("d1:", d1.shape)
        d1_uni = torch.cat([d1, f3_2], dim=1)
        d1_uni = self.decoder[3](d1_uni)        
        #print("d1_uni:", d1_uni.shape)

        d2 = self.decoder[4](d1_uni)
        #print("d2:", d2.shape)       
        d2_uni = torch.cat([d2, f2_2], dim=1)
        d2_uni = self.decoder[5](d2_uni)
        #print("d2_uni:", d2_uni.shape)
        
        d3 = self.decoder[6](d2_uni)
        #print("d3:", d3.shape)        
        d3_uni = torch.cat([d3, f1_2], dim=1)    
        d3_uni = self.decoder[7](d3_uni) 
        #print("d3_uni:", d3_uni.shape)        

        decoded = self.final_decoder(d3_uni)
        pixel_prob = self.pixel_prob_branch(d3_uni)

        return decoded, pixel_prob


class DCGAN_discriminator(nn.Module):
    def __init__(self):
        super(DCGAN_discriminator, self).__init__()
        nb_filters = 64
        nb_conv = int(np.floor(np.log(128) / np.log(2)))
        list_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

        self.conv1 = nn.Conv2d(3, list_filters[0], kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(list_filters[0])
        self.relu1 = nn.LeakyReLU(0.2)

        # Next convs
        conv_blocks = []
        for i, f in enumerate(list_filters[1:]):
            conv_name = "conv%d" % (i + 2)
            bn_name = "bn%d" % (i + 2)
            relu_name = "relu%d" % (i + 2)

            conv_block = nn.Sequential(
                nn.Conv2d(list_filters[i], f, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(f),
                nn.LeakyReLU(0.2)
            )

            setattr(self, conv_name, conv_block)
            conv_blocks.append(conv_name)

        self.conv_blocks = nn.Sequential(*[getattr(self, name) for name in conv_blocks])
        self.flatten = nn.Flatten()
        self.fc_out = nn.Linear(list_filters[-1] * 4 * 4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.conv_blocks(x)
        x = self.flatten(x)
        x = self.fc_out(x)
        x = self.sigmoid(x)
        return x
