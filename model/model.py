from model.blocks import double_conv, up_sample, down_sample
import torch
from torch import nn

class Unet_model_4(nn.Module):

    def __init__(self, input_layer):
        super().__init__()

        self.conv_1 = double_conv(input_layer, 8)
        self.conv_2 = double_conv(8, 16)
        self.conv_3 = double_conv(16, 32)
        self.conv_4 = double_conv(32, 64)
        self.conv_5 = double_conv(64, 128)

        self.decon_6 = up_sample(128, 64, bn=False)
        self.decon_7 = up_sample(64, 32, bn=False)
        self.decon_8 = up_sample(32, 16, bn=False)
        self.decon_9 = up_sample(16, 8, bn=False)

        self.conv_6 = double_conv(128, 64)
        self.conv_7 = double_conv(64, 32)
        self.conv_8 = double_conv(32, 16)
        self.conv_9 = double_conv(16, 8)

        self.final = nn.Conv2d(8, 1, 1)

    def down(self, x):

        x_shapes = x.shape[2], x.shape[3]
        if not x_shapes[0] % 2 == 0:
            x = nn.ZeroPad2d((0, 1, 0, 1))(x)
        if not x_shapes[1] % 2 == 0:
            x = nn.ZeroPad2d((0, 1, 0, 1))(x)

        return nn.MaxPool2d(2)(x)

    def concat(self, upscale, conv_x, n_filter):

        upscale_shape = upscale.shape
        conv_x_shape = conv_x.shape

        if (upscale_shape[2]-conv_x_shape[2]) >=0 and (upscale_shape[3]-conv_x_shape[3])>=0:

            offsets = [0, 0, (upscale_shape[2]-conv_x_shape[2])//2, (upscale_shape[3]-conv_x_shape[3])//2]
            size = [conv_x_shape[0], n_filter, conv_x_shape[2], conv_x_shape[3]]
            # offsets for the top left corner of the crop
            a1,a2,a3,a4 = offsets
            b1,b2,b3,b4 = size

            upscale = upscale[a1:a1 + b1, a2:a2 + b2, a3:a3 + b3, a4:a4 + b4]

        else:

            offsets = [0, 0, (conv_x_shape[2] - upscale_shape[2]) // 2, (conv_x_shape[3] - upscale_shape[3]) // 2]
            size = [upscale_shape[0], n_filter, upscale_shape[2], upscale_shape[3]]
            # offsets for the top left corner of the crop
            a1, a2, a3, a4 = offsets
            b1, b2, b3, b4 = size

            conv_x = conv_x[a1:a1 + b1, a2:a2 + b2, a3:a3 + b3, a4:a4 + b4]

        concat = torch.cat([upscale, conv_x], dim=1)

        return concat

    def forward(self, x):

        conv1 = self.conv_1(x)
        x = self.down(conv1)

        conv2 = self.conv_2(x)
        x = self.down(conv2)

        conv3 = self.conv_3(x)
        x = self.down(conv3)

        conv4 = self.conv_4(x)
        x = self.down(conv4)

        x = self.conv_5(x)

        x = self.decon_6(x)
        x = self.concat(x, conv4, 64)
        x = self.conv_6(x)

        x = self.decon_7(x)
        x = self.concat(x, conv3, 32)
        x = self.conv_7(x)

        x = self.decon_8(x)

        x = self.concat(x, conv2, 16)
        x = self.conv_8(x)

        x = self.decon_9(x)
        x = self.concat(x, conv1, 8)
        x = self.conv_9(x)

        out = self.final(x)
        x = torch.sigmoid(out)

        return x

