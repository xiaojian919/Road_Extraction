import torch
import torch.nn.functional as F
from Models.attention import CAMBlock,PAMBlock,Head


class CNNLayer(torch.nn.Module):
    def __init__(self, C_in, C_out):
        super(CNNLayer, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(C_in, C_out, 3, 1, 1),
            torch.nn.BatchNorm2d(C_out),
            torch.nn.Dropout(0.3),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(C_out, C_out, 3, 1, 1),
            torch.nn.BatchNorm2d(C_out),
            torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DownSampling(torch.nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(C, C, 3, 2, 1),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSampling(torch.nn.Module):
    def __init__(self, C):
        super(UpSampling, self).__init__()
        self.C = torch.nn.Conv2d(C, C // 2, 1, 1)#逐点卷积

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode='nearest')#临近插值
        x = self.C(up)
        return torch.cat((x, r), 1)


class CPA_u_net(torch.nn.Module):
    def __init__(self):
        super(CPA_u_net, self).__init__()
        self.C1 = CNNLayer(3, 64)
        self.D1 = DownSampling(64)#128
        self.C2 = CNNLayer(64, 128)
        self.D2 = DownSampling(128)#64
        self.C3 = CNNLayer(128, 256)
        self.D3 = DownSampling(256)#32
        self.C4 = CNNLayer(256, 512)
        self.D4 = DownSampling(512)#16
        self.C5 = CNNLayer(512, 1024)

        self.cam = CAMBlock(1024)
        self.pam = PAMBlock(1024)

        self.U1 = UpSampling(1024)#32
        self.C6 = CNNLayer(1024, 512)
        self.U2 = UpSampling(512)#64
        self.C7 = CNNLayer(512, 256)
        self.U3 = UpSampling(256)#128
        self.C8 = CNNLayer(256, 128)
        self.U4 = UpSampling(128)#256
        self.C9 = CNNLayer(128, 64)
        self.pre = torch.nn.Conv2d(64, 1, 3, 1, 1)#转换通道
        self.Th = torch.nn.Sigmoid()

    def forward(self, x):
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))#128
        R3 = self.C3(self.D2(R2))#64
        R4 = self.C4(self.D3(R3))#32
        Y1 = self.C5(self.D4(R4))#16

        att_Y1 = self.cam(Y1)
        att_Y1 = self.pam(att_Y1)

        O1 = self.C6(self.U1(att_Y1, R4))#32
        O2 = self.C7(self.U2(O1, R3))#64
        O3 = self.C8(self.U3(O2, R2))#128
        O4 = self.C9(self.U4(O3, R1))#256
        return self.Th(self.pre(O4))#[3,256,256]

if __name__ == '__main__':
    net = DA_u_net().cuda()
    # summary(net,(3,256,256))
    x = torch.ones([2, 3, 64, 64]).cuda()
    out = net(x)
    # pam = PAMBlock(512).cuda()
    # cam = CAMBlock(512).cuda()
    # pamout = pam(x)
    # print(pamout.shape)
    # out = cam(pamout)
    # head = Head(512, 512, nn.BatchNorm2d).cuda()
    # out = head(x)
    # print(out.shape)