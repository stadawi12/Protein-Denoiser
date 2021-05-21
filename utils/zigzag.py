import torch
import torch.nn as nn
import torch.nn.functional as F

def crop_img(tensor, target):
    """
        This function should crop the tensor to the size
        of the target,  it should crop out the centre of
        the tensor rather than a corner of it.
    """
    tensor_size = tensor.shape[2]
    target_size = target.shape[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, 
            delta:tensor_size - delta,
            delta:tensor_size - delta,
            delta:tensor_size - delta
            ]

# double convolution, frequently used inside unet
def double_conv(in_c, out_c):
    conv = nn.Sequential(
           nn.Conv3d(in_c , out_c, kernel_size=3, padding=1),
           nn.ReLU(inplace=True),
           nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
           nn.ReLU(inplace=True)
            )
    return conv

# The 3D UNet class
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Max Pool function, halves the input
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # All down convolutions, left side of unet
        self.down_conv1 = double_conv(  1,   64)
        self.down_conv2 = double_conv( 64,  128)
        self.down_conv3 = double_conv(128,  256)
        self.down_conv4 = double_conv(256,  512)
        self.down_conv5 = double_conv(512,  256)
        self.down_conv6 = double_conv(256,  128)
        self.down_conv7 = double_conv(128,  64)
        # self.down_conv5 = double_conv(512, 1024)

        # All transpose convolutions, right side of unet
        self.trans1 = nn.ConvTranspose3d(
                in_channels  = 512,
                out_channels = 256,
                kernel_size  = 2,
                stride       = 2
                )
        # self.trans2 = nn.ConvTranspose3d(
        #         in_channels  = 256,
        #         out_channels = 128,
        #         kernel_size  = 2,
        #         stride       = 2
        #         )
        self.trans3 = nn.ConvTranspose3d(
                in_channels  = 128,
                out_channels = 64,
                kernel_size  = 2,
                stride       = 2
                )
        self.trans4 = nn.ConvTranspose3d(
                in_channels  = 64,
                out_channels = 32,
                kernel_size  = 2,
                stride       = 2
                )

        # All convolutions but on the right side of unet
        # self.up_conv1 = double_conv(512, 256)
        # self.up_conv2 = double_conv(256, 128)
        # self.up_conv3 = double_conv(128,  64)
        # self.up_conv4 = double_conv(32 ,  16)

        # Final convolution to get the right output
        self.out1 = nn.Conv3d(
                    in_channels  = 32,
                    out_channels = 1,
                    kernel_size  = 1,
                    )
        # self.out2 = nn.Conv3d(
        #             in_channels  = 8,
        #             out_channels = 1,
        #             kernel_size  = 4
        #             )

    def forward(self, image):
        x1 = self.down_conv1(image) # conv down

        x2 = self.max_pool(x1)      # max pool
        x2 = self.down_conv2(x2)    # conv down

        x3 = self.trans3(x2)        # double
        x1 = crop_img(x1, x3)       # crop
        x3 = self.down_conv3(torch.cat([x3, x1], 1))
        print(f"shape of x3: {x3.shape}")

        x4 = self.max_pool(x3)
        x4 = self.down_conv4(x4)

        x5 = self.trans1(x4)
        x3 = crop_img(x3, x5)
        x5 = self.down_conv5(torch.cat([x5,x3],1))
        print(f"shape of x5: {x5.shape}")
        x5 = self.down_conv6(x5)
        x5 = self.down_conv7(x5)
        x5 = self.trans4(x5)
        x5 = self.out1(x5)

        return x5

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    unet = UNet()
    unet = unet.to(device)
    rnd = torch.randn(3,1,64,64,64)
    rnd = rnd.to(device)
    out = unet(rnd)
    pytorch_total_params = sum(p.numel() for p in unet.parameters())
    print(f"No. of params: {pytorch_total_params}")
    print(f"Shape of output: {out.shape}")
