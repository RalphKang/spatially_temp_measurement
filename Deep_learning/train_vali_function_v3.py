# %%
import torch
import torch.optim
import copy
import pandas as pd
import numpy as np


def get_lr(optimizer):
    # set original learning rate
    for param_group in optimizer.param_groups:
        return param_group['lr']


def change_lr(initial_lr, optimizer, ite, boundary=10, scale=20, boundary2=100, scale2=50, lr2=1e-4, mode='cosine'):
    if mode == "exp":
        step = (initial_lr - 1e-7) / boundary
        if ite < boundary:
            lr = 1e-7 + step * ite  # exponential  increasing learning rate for learning rate search
            print("warm_up")
        if ite >= boundary:
            lr = initial_lr * np.exp(-(ite - boundary) / scale)
    elif mode == "cosine":
        step = (initial_lr - 1e-7) / boundary
        if ite < boundary:
            lr = 1e-7 + step * ite  # exponential  increasing learning rate for learning rate search
            print("warm_up")
        if ite >= boundary:
            lr = (initial_lr - 1e-7) * np.abs(np.cos((ite - boundary) / scale * np.pi)) + 1e-7
    elif mode == "cos_exp":
        step = (initial_lr - 1e-7) / boundary
        if ite < boundary:
            lr = 1e-7 + step * ite  # exponential  increasing learning rate for learning rate search
            print("warm_up")
        elif boundary <= ite < 300:
            lr = (initial_lr - 1e-7) * np.abs(np.cos((ite - boundary) / scale * np.pi)) + 1e-7
        elif ite >= 300:
            lr = lr2 * np.exp(-(ite - boundary2) / scale2)
    else:
        lr = initial_lr
        print("constant")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warm_up_lr(initial_lr, optimizer, ite, boundary=10):
    step = (initial_lr - 1e-7) / boundary
    if ite < boundary:
        lr = 1e-7 + step * (ite+1)  # exponential  increasing learning rate for learning rate search
        print("warm_up")
    else:
        lr = initial_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def change_lr_search(optimizer, ite):
    lr = 1e-7 * 5 ** ite  # exponential  increasing learning rate for learning rate search
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_function(device,model, train_dl, optimizer, loss, ep, epoch, train_loss_record, sanity_check=False, lr_search=False):
    total_batch_train = len(train_dl)
    model.train()
    train_loss_sum = 0
    current_lr = get_lr(optimizer)
    print("current learning rate:{:.9f}".format(current_lr))
    for i, (input_train, target_train) in enumerate(train_dl):
        input_train = input_train.to(device)
        target_train =target_train.to(device)
        # forward calculation
        out_train_hat = model(input_train)
        loss_train = loss(out_train_hat, target_train)
        if lr_search:
            print('every batch loss={:.6f}'.format(loss_train))
            loss_batch_record = loss_train.to("cpu")
            old_lr = get_lr(optimizer)
            find_lr = np.array([old_lr, loss_batch_record.item()])
            find_lr_pd = pd.DataFrame(find_lr)
            find_lr_pd.to_csv("batch_loss.csv", mode='a')
            change_lr_search(optimizer, i)
            if old_lr >= 1.:  # if learning rate is too large, it's not meaningful
                break
        train_loss_sum += loss_train
        # backward propagation
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        if (i + 1) % 50 == 0:
            print("epoch number:{}/{},batch number:{}/{},loss:{:.4f}".format(ep + 1, epoch, i + 1, total_batch_train,
                                                                             loss_train.item()))
        if sanity_check:
            break
    average_loss_train = train_loss_sum / total_batch_train
    print('train loss of the model: {:.6f},epoch={}'.format(average_loss_train, ep + 1))
    train_loss_record.append(average_loss_train)
    return train_loss_record


def vali_function(device,model, model_save_dir, vali_dl, loss, ep, epoch, vali_loss_record, sanity_check=False,
                  historical_best=float('inf')):
    total_batch_eval = len(vali_dl)
    model.eval()
    vali_loss_sum = 0
    with torch.no_grad():
        for j, (input_eval, target_eval) in enumerate(vali_dl):
            input_eval = input_eval.to(device)
            target_eval = target_eval.to(device)
            out_eval_hat = model(input_eval)
            loss_eval = loss(out_eval_hat, target_eval)
            vali_loss_sum = vali_loss_sum + loss_eval
            if (j + 1) % 50 == 0:
                print(
                    "epoch number:{}/{},batch number:{}/{},loss:{:.4f}".format(ep + 1, epoch, j + 1, total_batch_eval,
                                                                               loss_eval.item()))
            if sanity_check:
                break
        average_loss = vali_loss_sum / total_batch_eval
        print('validation loss of the model: {:.6f},epoch={}'.format(average_loss, ep + 1))
        vali_loss_record.append(average_loss)

        if average_loss.item() < historical_best:
            weight_best = copy.deepcopy(model.state_dict())
            torch.save(weight_best, model_save_dir)
            print("best model has been saved")
            historical_best=average_loss.item()
    return vali_loss_record,historical_best
