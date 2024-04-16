import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

import transformer.Constants as Constants
import Utils

from preprocess.Dataset import get_dataloader
from transformer.Models import Transformer
from tqdm import tqdm


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)  # num_types

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train_new.pkl', 'train')
    # train_data = train_data[:1600]
    print('[Info] Loading dev data...')
    dev_data, _ = load_data(opt.data + 'dev_new.pkl', 'dev')
    # dev_data = dev_data[:200]
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test_new.pkl', 'test')
    # test_data = test_data[:200]

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    return trainloader, testloader, num_types


def train_epoch(model, training_data, optimizer, optimizer2, pred_loss_func, opt, epoch,
                other_params, paras, num_types, delta_matrix_pre):
    """ Epoch operation in training phase. """

    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    batch_i = 0
    total_grad_delay = 0
    total_grad_other = 0
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        batch_i += 1
        """ prepare data """
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
        event_type = event_type - 1  # when event starts from 1

        # """ forward """
        # if 0 < epoch <= 20 or 40 < epoch <= 60 or 80 < epoch <= 100:
        # if 0 < epoch <= 10 or 20 < epoch <= 30 or 40 < epoch <= 50 or 60 < epoch <= 70 or 80 < epoch <= 90:
        # if 0 < epoch <= 10 or 30 < epoch <= 40 or 60 < epoch <= 70 or 90 < epoch <= 100:
        if 0 < epoch <= 10:
            optimizer2.zero_grad()  # only THP
        else:
            optimizer.zero_grad()  # THP + Masker
        # optimizer.zero_grad()

        enc_out, prediction, delta_matrix = model(event_type, event_time, delta_matrix_pre, epoch, batch_i)

        """ backward """
        # negative log-likelihood
        event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
        event_loss = -torch.sum(event_ll - non_event_ll)

        # type prediction
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func)

        # time prediction
        se = Utils.time_loss(prediction[1], event_time)

        # delay matrix sparsity penalty loss
        l2_loss = 0

        # Matrix norm
        # for param in delta_matrix:
        #     l2_loss += torch.sum(torch.square(param))

        # L2,1 norm
        # for i in range(len(delta_matrix)):
        #     rows_norm = 0
        #     for param in delta_matrix[i]:
        #         rows_norm += torch.sum(torch.square(param))
        #     l2_loss += torch.sqrt(rows_norm)

        # L1,1 norm
        # for i in range(len(delta_matrix)):
        #     rows_norm = torch.sum(torch.abs(delta_matrix[i, :]))
        #     l2_loss += torch.abs(rows_norm)

        # L1,2 norm
        # for i in range(len(delta_matrix)):
        #     rows_norm = torch.sum(torch.abs(delta_matrix[i, :]))
        #     l2_loss += torch.square(rows_norm)
        # l2_loss = torch.sqrt(l2_loss)


        # gradient norm
        # grad_norm_delay = 0
        # grad_norm_other = 0
        # for name, param in model.named_parameters():
        #     # if "masker" in name:
        #     #     print(name, param, param.grad)
        #     if param.grad is not None:
        #         if "masker" in name:
        #             grad_norm_delay += param.grad.data.norm(2).item() ** 2
        #         else:
        #             grad_norm_other += param.grad.data.norm(2).item() ** 2
        # grad_norm_delay = grad_norm_delay ** 0.5
        # grad_norm_other = grad_norm_other ** 0.5

        # SE is usually large, scale it to stabilize training
        scale_time_loss = 100
        weight_decay = 0.01
        weight_decay_delay = 1
        lambda1 = 0.1
        all_linear1_params = torch.cat([x.view(-1) for x in model.masker.fc2.parameters()])
        l1_regularization = lambda1 * torch.norm(all_linear1_params, 1)  # l1 norm
        # loss = weight_decay * grad_norm_other + pred_loss + se / scale_time_loss + weight_decay_delay * grad_norm_delay
        loss = event_loss + pred_loss + se / scale_time_loss  # original
        # loss = event_loss + pred_loss + se / scale_time_loss + l1_regularization  # original
        # loss.backward()  # original
        loss.backward(retain_graph=True)

        # torch.autograd.set_detect_anomaly(True)
        # with torch.autograd.detect_anomaly():
        #     loss.backward()

        """ update parameters """
        # gradient norm
        grad_norm_delay = 0
        grad_norm_other = 0
        for name, param in model.named_parameters():
            # if "masker" in name:
            #     print(name, param, param.grad)
            if param.grad is not None:
                if "masker" in name:
                    grad_norm_delay += param.grad.data.norm(2).item() ** 2
                else:
                    grad_norm_other += param.grad.data.norm(2).item() ** 2
        grad_norm_delay = grad_norm_delay ** 0.5
        grad_norm_other = grad_norm_other ** 0.5
        # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        total_loss = loss + weight_decay_delay * grad_norm_delay + weight_decay * grad_norm_other
        # print(loss, grad_norm_delay, grad_norm_other, total_loss)
        total_loss.backward()

        # for name, parms in model.named_parameters():
        #     if "delta_matrix" in name:
        #         delta_matrix_grad = np.round(parms.grad.detach().numpy(), 3)
        # delta_matrix_np = np.round(delta_matrix.detach().numpy(), 3)
        # with open(opt.log, 'a') as f:
        #     f.write("delay:" + "\n")
        #     for line in delta_matrix_np:
        #         f.write("".join(str(line)) + "\n")
        #     f.write("gradient:" + "\n")
        #     for line in delta_matrix_grad:
        #         f.write("".join(str(line)) + "\n")

        # if 0 < epoch <= 20 or 40 < epoch <= 60 or 80 < epoch <= 100:
        # if 0 < epoch <= 10 or 20 < epoch <= 30 or 40 < epoch <= 50 or 60 < epoch <= 70 or 80 < epoch <= 90:
        # if 0 < epoch <= 10 or 30 < epoch <= 40 or 60 < epoch <= 70 or 90 < epoch <= 100:
        if 0 < epoch <= 10:
            optimizer2.step()  # only THP
        else:
            optimizer.step()  # THP + Masker
        # optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

        total_grad_delay += grad_norm_delay
        total_grad_other += grad_norm_other

    rmse = np.sqrt(total_time_se / total_num_pred)
    return (total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse, delta_matrix,
            total_grad_delay / batch_i, total_grad_other / batch_i)


def eval_epoch(model, validation_data, pred_loss_func, delta_matrix_pre, num_types, opt, epoch):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    # total_grad_delay = 0
    # total_grad_other = 0
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
            event_type = event_type - 1  # when event starts from 1
            """ forward """
            enc_out, prediction, delta_matrix = model(event_type, event_time, delta_matrix_pre, epoch, batch_i=0)
            # print("validata delay: ", delta_matrix)
            """ compute loss """
            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
            event_loss = -torch.sum(event_ll - non_event_ll)
            _, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
            se = Utils.time_loss(prediction[1], event_time)

            """ note keeping """
            # gradient norm
            # grad_norm_delay = 0
            # grad_norm_other = 0
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         if "masker" in name:
            #             grad_norm_delay += param.grad.data.norm(2).item() ** 2
            #         else:
            #             grad_norm_other += param.grad.data.norm(2).item() ** 2
            # grad_norm_delay = grad_norm_delay ** 0.5
            # grad_norm_other = grad_norm_other ** 0.5
            # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # print(grad_norm_delay, grad_norm_other, grad_norm)

            total_event_ll += -event_loss.item()
            total_time_se += se.item()
            total_event_rate += pred_num.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]
            # total_grad_delay += grad_norm_delay
            # total_grad_other += grad_norm_other

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse

def train(model, training_data, validation_data, optimizer, optimizer2, scheduler, scheduler2, pred_loss_func, opt,
          other_params, paras, num_types):
    """ Start training. """

    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE
    delta_matrix_pre = torch.zeros(num_types)
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_time, delta_matrix, train_delay, train_other = train_epoch(model, training_data, optimizer, optimizer2, pred_loss_func, opt, epoch,
                                                                        other_params, paras, num_types, delta_matrix_pre)
        delta_matrix_pre = delta_matrix.detach()
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min, '
              .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))
        delta_matrix_np = np.round(delta_matrix.detach().numpy(), 3)
        print("delta_matrix in Main train: \n", delta_matrix_np)

        # for name, parms in model.named_parameters():
        #     if "masker" in name:
        #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
        #               ' -->grad_value:', parms.grad)
        #         delta_matrix_grad = np.round(parms.grad.detach().numpy(), 3)
        print("train delay: ", train_delay)
        print("train other: ", train_other)

        start = time.time()
        valid_event, valid_type, valid_time = eval_epoch(model, validation_data, pred_loss_func, num_types, delta_matrix_pre, opt, epoch)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

        valid_event_losses += [valid_event]
        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]
        print('  - [Info] Maximum ll: {event: 8.5f}, '
              'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
              .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse)))


        # logging
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}, {train_ll: 8.5f}, {train_delay: 8.5f}, {train_other: 8.5f}\n'
                    .format(epoch=epoch, ll=valid_event, acc=valid_type, train_ll=train_event, rmse=valid_time, train_delay=train_delay, train_other=train_other))
            f.write("delay:" + "\n")
            for line in delta_matrix_np:
                f.write("".join(str(line)) + "\n")
            # f.write("gradient:" + "\n")
            # for line in delta_matrix_grad:
            #     f.write("".join(str(line)) + "\n")

        # if 0 < epoch <= 20 or 40 < epoch <= 60 or 80 < epoch <= 100:
        # if 0 < epoch <= 10 or 20 < epoch <= 30 or 40 < epoch <= 50 or 60 < epoch <= 70 or 80 < epoch <= 90:
        # if 0 < epoch <= 10 or 30 < epoch <= 40 or 60 < epoch <= 70 or 90 < epoch <= 100:
        if 0 < epoch <= 10:
            scheduler2.step()  # only THP
        else:
            scheduler.step()  # THP + Masker
        # scheduler.step()
        # scheduler2.step()

        # print("saving model")
        # torch.save(model, opt.data + "/lr_delay_" + str(opt.lr) + "/" + str(epoch) + "model.pt")

def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=16)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-smooth', type=float, default=0.1)

    parser.add_argument('-log', type=str, default='log.txt')

    parser.add_argument('-num_events', type=int, default=32)
    parser.add_argument('-lr_delay', type=float, default=0.01)
    parser.add_argument('-n_hidden', type=int, default=16)
    parser.add_argument('-n_input', type=int, default=3)

    opt = parser.parse_args()

    # default device is CUDA
    # opt.device = torch.device('cuda')
    opt.device = torch.device('cpu')

    # setup the log file
    with open(opt.log, 'w') as f:
        f.write(format(opt))
        f.write('\n')
        f.write('Epoch, Log-likelihood-Test, Accuracy, RMSE, Log-likelihood-Train, Gradient_Masker, Gradient_THP\n')

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """
    trainloader, testloader, num_types = prepare_dataloader(opt)
    """ prepare model """
    model = Transformer(
        epoch_num=opt.epoch,
        num_types=num_types,
        num_events=opt.num_events,
        d_model=opt.d_model,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
        n_hidden=opt.n_hidden,
        n_input=opt.n_input
    )
    model.to(opt.device)

    """ optimizer and scheduler """
    print(model)
    paras = []
    for name, p in model.named_parameters():
        if "masker" in name:
            p.requires_grad = True
            paras.append(p)

    # optimizer2 = optim.Adam(paras, opt.lr_delay, betas=(0.9, 0.999), eps=1e-05)
    # scheduler2 = optim.lr_scheduler.StepLR(optimizer2, 10, gamma=0.5)

    all_params = model.parameters()
    params_id = list(map(id, paras))
    other_params = list(filter(lambda p: p.requires_grad and id(p) not in params_id, all_params))
    optimizer = optim.Adam([
        {'params': other_params},
        {'params': paras, 'lr': opt.lr_delay}],
        opt.lr, betas=(0.9, 0.999), eps=1e-05
    )

    # no delay
    optimizer2 = optim.Adam([
        {'params': other_params}],
        opt.lr, betas=(0.9, 0.999), eps=1e-05
    )
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, 10, gamma=0.5)

    # original
    # optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
    #                        opt.lr, betas=(0.9, 0.999), eps=1e-05)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)



    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    train(model, trainloader, testloader, optimizer, optimizer2, scheduler, scheduler2, pred_loss_func, opt,
          other_params, paras, num_types)

    print("saving model")
    torch.save(model, opt.data + "model.pt")

if __name__ == '__main__':
    main()
