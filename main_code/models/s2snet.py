import torch
from torch import nn
from torch.autograd import Variable
from main_code.models.backbone import vgg
import torch.nn.functional as F
import os

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding,  bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * F.relu(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * F.relu(cc)
        return ch, cc

    def init_hidden(self, hidden, shape):
        self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()



class ConvLSTM(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.bias = bias

        self.cell = ConvLSTMCell(self.input_channels, self.hidden_channels, self.kernel_size, self.bias)


    def forward(self,seqs,h0,c0):

        outputs = []
        steps = len(seqs)
        h_pre = h0
        c_pre = c0
        for i in range(steps):
            x = seqs[i]

            if i==0:
                bsize, _, height, width = x.size()
                self.cell.init_hidden(hidden=self.hidden_channels,shape=(height, width))

            h_new, c_new = self.cell(x, h_pre, c_pre)
            h_pre = h_new
            c_pre = c_new


            outputs.append(h_pre)

        return outputs


class S2SNET(nn.Module):
    def __init__(self):
        super(S2SNET, self).__init__()

        self.init_backbone = vgg.vgg16_bn(pretrained=True)
        self.encode_backbone = vgg.vgg16_bn(pretrained=True)


        self.init_backbone.features[0] = nn.Conv2d(4,64,kernel_size=3,padding=1)
        self.initializer = nn.Sequential(
            self.init_backbone.features,
            nn.Conv2d(512,4096,kernel_size=1),
            nn.ReLU(True)
        )

        self.init_c0conv = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=1),
            nn.ReLU(True)
        )
        self.init_h0conv = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=1),
            nn.ReLU(True)
        )


        self.encoder = nn.Sequential(
            self.encode_backbone.features,
            nn.Conv2d(512,4096,kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(4096, 512, kernel_size=1),
            nn.ReLU(True)
        )

        self.convLSTM = ConvLSTM(input_channels=512,hidden_channels=512,kernel_size=3)


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2),

        )


    def forward(self, init_x, seq_x):
        bs,ch,h,w = init_x.size()
        seqs = torch.split(seq_x,1,dim=1)

        encoder_features = []
        for seq in seqs:
            seq = torch.squeeze(seq,1)
            en_feature = self.encoder(seq)
            encoder_features.append(en_feature)

        init_x = self.initializer(init_x)
        h0 = self.init_h0conv(init_x)
        c0 = self.init_c0conv(init_x)

        convlstm_outs = self.convLSTM(encoder_features,h0,c0)

        decoder_outs = []
        for lstm_out in convlstm_outs:
            de_out = self.decoder(lstm_out)
            de_out = self.crop_feature(de_out,h,w)
            de_out = torch.unsqueeze(de_out,1)
            decoder_outs.append(de_out)

        logits = torch.cat(decoder_outs,1)

        return logits

    def crop_feature(self,feature,h,w):

        _,_,oh,ow = feature.size()

        sh = (oh-h)//2
        sw = (ow-w)//2

        out_feature = feature[:,:,sh:sh+h,sw:sw+w]

        return out_feature

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = S2SNET().cuda()

    init_x = Variable(torch.randn(2, 4, 256, 448)).cuda()
    seq_x = Variable(torch.randn(2, 8, 3, 256, 448)).cuda()
    target = Variable(torch.randn(2,8, 1, 256, 448)).double().cuda()

    logits = model(init_x,seq_x)

    print(logits.size())
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    # res = torch.autograd.gradcheck(loss_fn, (logits.double(), target), eps=1e-6, raise_exception=True)
    # print(res)
























