# %%
from sys import exit
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader, TensorDataset
# from dataset_batch import *
from train_vali_function_v3 import train_function, vali_function, get_lr, warm_up_lr
from algorithm.xception import *
from algorithm.vgg import *
from algorithm.resnet import *
from algorithm.inceptionv3 import *
from algorithm.shufflenet import *
from algorithm.squeezenet import *
from sys import exit
import gc

def main(guide_lr, guide_wd, epoch_size, smp_dir, tgt_dir, pretrain_mode, model_name):
    dev = "cuda"
    search_lr = True
    # set seed for numpy and pytorch
    torch.manual_seed(1)
    np.random.seed(1)

    # %% data_set and data_load
    # %% data_reading
    data = pd.read_csv(smp_dir, header=None, dtype=np.float32)
    target = pd.read_csv(tgt_dir, header=None, dtype=np.float32)
    # %% dimension augmentation
    data_np = np.array(data)
    # %% sampling
    data_new = data_np[:, ::6]
    data_need = data_new[:, 0:1024]
    target_np = np.array(target)
    max_data = np.max(data_need, axis=0)
    min_data = np.min(data_need, axis=0)
    data_scaled = (data_need - min_data) / (max_data - min_data)
    data_np_ext2 = data_scaled.reshape(-1, 1, 32, 32)
    # %% change to dataset
    sample_tc = torch.from_numpy(data_np_ext2)
    sample_tcc = sample_tc.float()
    label_tc = torch.from_numpy(target_np)
    label_tcc = label_tc.float()
    data_set_all = TensorDataset(sample_tcc, label_tcc)

    # %% feed to dataloader( data generator)
    data_length = len(data_set_all)
    train_length = int(0.85 * data_length)
    vali_length = data_length - train_length
    train_set, vali_set, = random_split(data_set_all, [train_length, vali_length])
    train_dl = DataLoader(train_set, batch_size=32, drop_last=True)
    vali_dl = DataLoader(vali_set, batch_size=32, drop_last=True)
    # %% # del unnecessary parameters
    del smp_dir, tgt_dir, data, target, data_np, target_np, max_data, min_data, data_scaled, data_np_ext2
    del sample_tc, sample_tcc, label_tcc, label_tc, data_set_all, data_length, vali_length, train_set, vali_set
    gc.collect()
    # %% **********************compile model**********************************************
    model_name_list = ["xception", 'inceptionv3', 'resnet18', 'resnet34', 'resnet50', 'vgg11',
                       'vgg13', 'vgg16', 'vgg19', "shufflenet", 'squeezenet']
    assert model_name in model_name_list
    if model_name == "xception":
        model = Xception(MiddleFLowBlock)
    if model_name == "inceptionv3":
        model = InceptionV3()
    if model_name == "vgg11":
        model = VGG(make_layers(cfg['A'], batch_norm=False))
    if model_name == "vgg13":
        model = VGG(make_layers(cfg['B'], batch_norm=False))
    if model_name == "vgg16":
        model = VGG(make_layers(cfg['D'], batch_norm=False))
    if model_name == "vgg19":
        model = VGG(make_layers(cfg['E'], batch_norm=False))
    if model_name == "resnet18":
        model = ResNet(BasicBlock, [2, 2, 2, 2])
    if model_name == "resnet34":
        model = ResNet(BasicBlock, [3, 4, 6, 3])
    if model_name == "resnet50":
        model = ResNet(BottleNeck, [3, 4, 6, 3])
    if model_name == "shufflenet":
        model = ShuffleNet([4, 8, 4])
    if model_name == "squeezenet":
        model = SqueezeNet(class_num=11)
    model.to(dev)
    model_save_dir = './model/' + model_name + '.pt'
    performance_dir = './model/' + model_name + '.txt'

    # lr setting---------------------------------------------------------------------------
    initial_lr = 1e-5  # no need to change for pretrain
    if pretrain_mode:
        model.load_state_dict(torch.load(model_save_dir))
        history = np.loadtxt(performance_dir)
        initial_lr = history[-1, 0]  # find the latest lr rate
        historical_best = history[-1, 2]
        print("historical best obtained is {:.6f}".format(historical_best))
    else:
        historical_best = float('inf')
    if guide_lr > 0.0:
        initial_lr = guide_lr
    if search_lr:
        initial_lr = 1e-8
        pretrain_mode = False
    optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=guide_wd)
    lr_scd = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    loss = nn.MSELoss(reduction="mean")
    # training setting---------------------------------------------------------------------
    epoch0 = 10  # warmup
    epoch = epoch_size  # for official training

    train_loss_record = []
    vali_loss_record = []
    lr_record = []
    # %% train and validation model
    # -------------warm up training-----------------------------------------
    if not pretrain_mode:
        for ep in range(epoch0):
            warm_up_lr(initial_lr=initial_lr, optimizer=optimizer, ite=ep, boundary=epoch0)
            train_loss_record = train_function(device=dev, model=model, train_dl=train_dl, optimizer=optimizer,
                                               loss=loss, ep=ep,
                                               epoch=epoch0,
                                               train_loss_record=train_loss_record, lr_search=search_lr)
            if search_lr:
                exit()
            vali_loss_record, historical_best = vali_function(device=dev, model=model, model_save_dir=model_save_dir,
                                                              vali_dl=vali_dl,
                                                              loss=loss, ep=ep,
                                                              epoch=epoch0, vali_loss_record=vali_loss_record,
                                                              historical_best=historical_best)
            # vital parameters record
            old_lr = get_lr(optimizer)
            lr_record.append(old_lr)
            f = open(performance_dir, 'a')  # open file in append mode
            np.savetxt(f, np.c_[
                old_lr, train_loss_record[-1].cpu().detach().numpy(), vali_loss_record[-1].cpu().detach().numpy()])
            f.close()
    # --------------------official training----------------------------------------------------
    for ep in range(epoch):
        # change_lr(initial_lr,optimizer=optimizer,ite=ep,mode="exp",scale=30)
        print("official training")
        train_loss_record = train_function(device=dev, model=model, train_dl=train_dl, optimizer=optimizer, loss=loss,
                                           ep=ep,
                                           epoch=epoch, train_loss_record=train_loss_record, lr_search=False)
        vali_loss_record, historical_best = vali_function(device=dev, model=model, model_save_dir=model_save_dir,
                                                          vali_dl=vali_dl,
                                                          loss=loss, ep=ep, epoch=epoch,
                                                          vali_loss_record=vali_loss_record,
                                                          historical_best=historical_best)
        old_lr = get_lr(optimizer)
        lr_scd.step()
        lr_record.append(old_lr)
        f = open(performance_dir, 'a')  # open file in append mode
        np.savetxt(f, np.c_[
            old_lr, train_loss_record[-1].cpu().detach().numpy(), vali_loss_record[-1].cpu().detach().numpy()])
        f.close()


if __name__ == '__main__':
    smp_dir = "./dataset/train_data.csv"
    tgt_dir = "./dataset/train_temp_normalized.csv"
    main(guide_lr=1e-5, guide_wd=1e-4, epoch_size=1000, smp_dir=smp_dir, tgt_dir=tgt_dir,
         pretrain_mode=False, model_name='vgg13')
