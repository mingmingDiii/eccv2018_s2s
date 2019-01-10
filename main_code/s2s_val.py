import torch
from torch.utils import data
from torch import nn
from torch import optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from main_code.models import s2snet
import matplotlib.pyplot as plt
from main_code.data_lib import ytbvos_dataset


import os
from tqdm import tqdm
import numpy as np

def _build_network(snapshot):
    epoch = 0
    net = s2snet.S2SNET()
    net = nn.DataParallel(net)
    if snapshot is not None:
        epoch = os.path.basename(snapshot).split('_')[2]
        epoch = int(epoch)+1
        net.load_state_dict(torch.load(snapshot))

    net = net.cuda()
    return net, epoch



os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def m_train(dataset_name='ytbyos',ex_='s2s'):



    dataset = ytbvos_dataset.YTBVOS_DATASET(mode='valid')



    # get data loader
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=4,
                                  pin_memory=True)


    # get net

    snapshot = '/mnt/sda1/don/documents/public/eccv2018_s2s/main_code/logs/s2s_ytbyos/weights/s2s_ytbyos_72'
    net,start_epoch = _build_network(snapshot)




    with torch.no_grad():
        train_iterator = tqdm(data_loader, total=len(data_loader))
        net.eval()

        for init_image_label,img_seqs in train_iterator:

            init_image_label_v = Variable(init_image_label).cuda()
            img_seqs_v = Variable(img_seqs).cuda()



            logit = net(init_image_label_v, img_seqs_v)

            pred = torch.sigmoid(logit)
            pred = pred.data.cpu().numpy()

            pred = np.where(pred>0.5,1.0,0.0)

            init_data = init_image_label.numpy()
            img_seqs_data = img_seqs.numpy()
            a = min(5,pred.shape[1])
            for i in range(a):
                plt.figure(figsize=(15,15))
                plt.subplot(211)
                plt.imshow(init_data[0,0,:,:])
                plt.imshow(init_data[0, -1, :, :],alpha=0.5)

                plt.subplot(212)
                plt.imshow(img_seqs_data[0,i,0,:,:])
                plt.imshow(pred[0,i,0,:,:],alpha=0.8)

                plt.show()





    pass




if __name__ == '__main__':

    m_train()











