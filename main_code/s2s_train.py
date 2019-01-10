import torch
from torch.utils import data
from torch import nn
from torch import optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from main_code.models import s2snet

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



    dataset = ytbvos_dataset.YTBVOS_DATASET(mode='train')


    ex_name = '{}_{}'.format(ex_,dataset_name)
    exp_dir = 'logs/{}/'.format(ex_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    ### weights path
    weights_path = exp_dir+'weights/'
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    ### board path
    board_path = exp_dir+'board/'
    if not os.path.exists(board_path):
        os.mkdir(board_path)
    writer = SummaryWriter(board_path)
    ### config file
    config_path = exp_dir+'config/'
    if not os.path.exists(config_path):
        os.mkdir(config_path)


    # get data loader
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)


    # get net

    snapshot = '/mnt/sda1/don/documents/public/eccv2018_s2s/main_code/logs/s2s_ytbyos/weights/s2s_ytbyos_58'
    net,start_epoch = _build_network(snapshot)

    # start_epoch = 0

    epochs = 80
    lr = 1e-5#config.learning_rate


    #criterion = class_balanced_cross_entropy_loss()
    criterion = nn.BCEWithLogitsLoss()
    val_loss = 0.0


    for epoch in range(start_epoch,start_epoch+epochs):
        # if epoch>=15:
        #     lr=1e-5
        if epoch>=60:
            lr=1e-6

        optimizer = optim.Adam(net.parameters(), lr=lr)
        epoch_train_loss = 0.0

        train_iterator = tqdm(data_loader, total=len(data_loader))
        net.train()
        steps = 0
        for init_image_label,img_seqs,ano_seqs in train_iterator:
            steps += 1
            init_image_label = Variable(init_image_label).cuda()
            img_seqs = Variable(img_seqs).cuda()
            ano_seqs = Variable(ano_seqs).cuda()


            optimizer.zero_grad()

            logit = net(init_image_label, img_seqs)

            train_loss = criterion(logit, ano_seqs)




            epoch_train_loss+=train_loss.data


            status = "[{}][{:03d}]" \
                     "la = {:0.7f}," \
                     "LR = {:0.7f} " \
                     "vall = {:0.5f},".format(
                ex_name, epoch,
                epoch_train_loss/steps,
                lr,
                val_loss)

            train_iterator.set_description(status)

            train_loss.backward()
            optimizer.step()


            # if steps%1000==0:
            #     torch.save(net.state_dict(), weights_path + '{}_{}_{}'.format(ex_name, epoch,steps))



        torch.save(net.state_dict(), weights_path + '{}_{}'.format(ex_name,epoch))



    pass




if __name__ == '__main__':

    m_train()











