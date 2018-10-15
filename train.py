# -*- coding: utf-8 -*-
# Author: Ji Yang <jiyang.py@gmail.com>
# License: MIT

import os
import time

import pandas as pd
import numpy as np

from common import iou, accuracy, FocalLoss2d

import torch
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def mkdir(dir_path):
    try:
        os.listdir(dir_path)
    except:
        os.mkdir(dir_path)


def cos_annealing_lr(initial_lr, cur_epoch, epoch_per_cycle):
    return initial_lr * (np.cos(np.pi * cur_epoch / epoch_per_cycle) + 1) / 2


def train_stage1(net, optimizer, train_loader, val_loader,
                 fold=None,
                 weight_path='stage_1_weights',
                 epochs=30):
    mkdir(weight_path)

    log = open('log{}.txt'.format(fold), 'a+')

    stat = pd.DataFrame(columns=['Training Loss', 'Training Acc', 'Training IoU',
                                 'Valid Loss', 'Valid Acc', 'Valid IoU'])

    best_val_iou = .0

    train_loss_record, train_acc_record, train_iou_record = [], [], []
    val_loss_record, val_acc_record, val_iou_record = [], [], []

    # criterion = torch.nn.BCELoss()
    criterion = FocalLoss2d()
    start = time.time()

    print(f'\nFold-{fold} Warm-up Training Overview')
    print('|          Train           |             Val          |')
    print('|-----------------------------------------------------|')
    print('|  Loss   |   Acc  |  IoU  |  Loss   |   Acc  |  IoU  |')
    log.write(f'\nFold-{fold} Warm-up Training Overview\n')
    log.write('|          Train           |             Val          |\n')
    log.write('|-----------------------------------------------------|\n')
    log.write('|  Loss   |   Acc  |  IoU  |  Loss   |   Acc  |  IoU  |\n')
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        epoch_iou = 0

        for batch_idx, batch in enumerate(train_loader):
            data, target = batch['image'].to(device), batch['mask'].to(device)
            output = net(data)
            # add sigmoid to the logits
            loss = criterion(output, target)
            acc = accuracy(output, target)
            iou_ = iou(output, target.int())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item() / len(train_loader)
            epoch_acc += acc.item() / len(train_loader)
            epoch_iou += iou_.item() / len(train_loader)

        print(f'|{epoch_loss:9.4f}|{epoch_acc:8.4f}|{epoch_iou:7.4f}', end='')
        log.write(f'|{epoch_loss:9.4f}|{epoch_acc:8.4f}|{epoch_iou:7.4f}')
        train_loss_record.append(epoch_loss)
        train_acc_record.append(epoch_acc)
        train_iou_record.append(epoch_iou)

        with torch.no_grad():
            epoch_val_loss = 0
            epoch_val_acc = 0
            epoch_val_iou = 0
            for batch_idx, batch in enumerate(val_loader):
                data, target = batch['image'].to(device), batch['mask'].to(device)

                output = net(data)
                loss = criterion(output, target)
                acc = accuracy(output, target)
                iou_ = iou(output, target.int())

                epoch_val_loss += loss.item() / len(val_loader)
                epoch_val_acc += acc.item() / len(val_loader)
                epoch_val_iou += iou_.item() / len(val_loader)

            print(f'|{epoch_val_loss:9.4f}|{epoch_val_acc:8.4f}|{epoch_val_iou:7.4f}|')
            log.write(f'|{epoch_val_loss:9.4f}|{epoch_val_acc:8.4f}|{epoch_val_iou:7.4f}|\n')
            val_loss_record.append(epoch_val_loss)
            val_acc_record.append(epoch_val_acc)
            val_iou_record.append(epoch_val_iou)

            if epoch_val_iou > best_val_iou:
                best_val_iou = epoch_val_iou
                torch.save(net.state_dict(), f'{weight_path}/Stage_1_Fold_{fold}')

    # stat['Epoch'] = [_ for _ in range(1, epochs + 1)]
    # stat['Stage'] = 1
    # stat['Fold'] = fold
    # stat['Training Loss'] = train_loss_record
    # stat['Training Acc'] = train_acc_record
    # stat['Training IoU'] = train_iou_record
    # stat['Valid Loss'] = val_loss_record
    # stat['Valid Acc'] = val_acc_record
    # stat['Valid IoU'] = val_iou_record
    print('Time used', time.time() - start)
    return net, stat


def finetune_stage1(net, train_loader, val_loader,
                    fold=None,
                    weight_path='finetune_stage_1_weights',
                    epochs=80):
    mkdir(weight_path)

    log = open('log{}.txt'.format(fold), 'a+')
    net.load_state_dict(torch.load(f'stage_1_weights/Stage_1_Fold_{fold}'))
    print(f'\nModel weights from stage_1_weights/Stage_1_Fold_{fold} Loaded')
    log.write(f'\nModel weights from stage_1_weights/Stage_1_Fold_{fold} Loaded\n')

    stat = pd.DataFrame(columns=['Training Loss', 'Training Acc', 'Training IoU',
                                 'Valid Loss', 'Valid Acc', 'Valid IoU'])

    best_val_iou = .0

    train_loss_record, train_acc_record, train_iou_record = [], [], []
    val_loss_record, val_acc_record, val_iou_record = [], [], []

    start = time.time()

    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)

    print(f'\nFold-{fold} Finetune Training Overview')
    print('|          Train           |             Val          |')
    print('|-----------------------------------------------------|')
    print('|  Loss   |   Acc  |  IoU  |  Loss   |   Acc  |  IoU  |')
    log.write(f'\nFold-{fold} Finetune Training Overview\n')
    log.write('|          Train           |             Val          |\n')
    log.write('|-----------------------------------------------------|\n')
    log.write('|  Loss   |   Acc  |  IoU  |  Loss   |   Acc  |  IoU  |\n')
    early_stop_counter = 0
    for epoch in range(epochs):
        if early_stop_counter > 15:
            print(f'\nEarly stop at epoch {epoch}!')
            log.write(f'\nEarly stop at epoch {epoch}!\n')
            break
        epoch_loss = 0
        epoch_acc = 0
        epoch_iou = 0

        for batch_idx, batch in enumerate(train_loader):
            data, target = batch['image'].to(device), batch['mask'].to(device)
            output = net(data)
            # add sigmoid to the logits
            loss = net.criterion(output, target)
            acc = accuracy(output, target)
            iou_ = iou(output, target.int())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item() / len(train_loader)
            epoch_acc += acc.item() / len(train_loader)
            epoch_iou += iou_.item() / len(train_loader)

        print(f'|{epoch_loss:9.4f}|{epoch_acc:8.4f}|{epoch_iou:7.4f}', end='')
        log.write(f'|{epoch_loss:9.4f}|{epoch_acc:8.4f}|{epoch_iou:7.4f}')
        train_loss_record.append(epoch_loss)
        train_acc_record.append(epoch_acc)
        train_iou_record.append(epoch_iou)

        with torch.no_grad():
            epoch_val_loss = 0
            epoch_val_acc = 0
            epoch_val_iou = 0
            for batch_idx, batch in enumerate(val_loader):
                data, target = batch['image'].to(device), batch['mask'].to(device)

                output = net(data)
                loss = net.criterion(output, target)
                acc = accuracy(output, target)
                iou_ = iou(output, target.int())

                epoch_val_loss += loss.item() / len(val_loader)
                epoch_val_acc += acc.item() / len(val_loader)
                epoch_val_iou += iou_.item() / len(val_loader)

            # Learning rate decay step
            scheduler.step(epoch_val_iou)
            print(f'|{epoch_val_loss:9.4f}|{epoch_val_acc:8.4f}|{epoch_val_iou:7.4f}|')
            log.write(f'|{epoch_val_loss:9.4f}|{epoch_val_acc:8.4f}|{epoch_val_iou:7.4f}|\n')
            val_loss_record.append(epoch_val_loss)
            val_acc_record.append(epoch_val_acc)
            val_iou_record.append(epoch_val_iou)

            if epoch_val_iou > best_val_iou:
                early_stop_counter = 0
                best_val_iou = epoch_val_iou
                torch.save(net.state_dict(), f'{weight_path}/Stage_2_Fold_{fold}')
            else:
                early_stop_counter += 1

    # stat['Epoch'] = [_ for _ in range(1, epochs + 1)]
    # stat['Stage'] = 1
    # stat['Fold'] = fold
    # stat['Training Loss'] = train_loss_record
    # stat['Training Acc'] = train_acc_record
    # stat['Training IoU'] = train_iou_record
    # stat['Valid Loss'] = val_loss_record
    # stat['Valid Acc'] = val_acc_record
    # stat['Valid IoU'] = val_iou_record
    print(f'Best --> {best_val_iou}')
    log.write(f'Best --> {best_val_iou}\n')
    print('Time used', time.time() - start)
    return net, stat


def finetune_stage2(net, train_loader, val_loader,
                    fold=None,
                    weight_path='finetune_stage_2_weights',
                    n_cycle=6,
                    initial_lr=0.01,
                    min_lr=0.001,
                    epochs_per_cycle=50):
    mkdir(f'{weight_path}_fold{fold}')

    log = open('log{}.txt'.format(fold), 'a+')
    net.load_state_dict(torch.load(f'finetune_stage_1_weights/Stage_2_Fold_{fold}'))
    print(f'\nModel weights from finetune_stage_1_weights/Stage_2_Fold_{fold} Loaded')
    log.write(f'\nModel weights from finetune_stage_1_weights/Stage_2_Fold_{fold} Loaded\n')

    lr_record = []
    train_loss_record, train_acc_record, train_iou_record = [], [], []
    val_loss_record, val_acc_record, val_iou_record = [], [], []

    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9, weight_decay=0.0001)

    for cycle in range(n_cycle):
        start = time.time()
        print(f'\nFold # {fold} Snapshot # {cycle} Overview')
        print('|          Train           |             Val          |')
        print('|-----------------------------------------------------|')
        print('|  Loss   |   Acc  |  IoU  |  Loss   |   Acc  |  IoU  |')
        log.write(f'\nFold # {fold} Snapshot # {cycle} Overview\n')
        log.write('|          Train           |             Val          |\n')
        log.write('|-----------------------------------------------------|\n')
        log.write('|  Loss   |   Acc  |  IoU  |  Loss   |   Acc  |  IoU  |\n')

        best_val_iou = .0
        for epoch in range(epochs_per_cycle):
            epoch_loss = 0
            epoch_acc = 0
            epoch_iou = 0

            lr = cos_annealing_lr(initial_lr, epoch, epochs_per_cycle)
            if lr < min_lr:
                break
            lr_record.append(lr)
            optimizer.state_dict()['param_groups'][0]['lr'] = lr

            for batch_idx, batch in enumerate(train_loader):
                data, target = batch['image'].to(device), batch['mask'].to(device)

                output = net(data)
                loss = net.criterion(output, target.long())

                acc = accuracy(output, target)
                iou_ = iou(output, target.int())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item() / len(train_loader)
                epoch_acc += acc.item() / len(train_loader)
                epoch_iou += iou_.item() / len(train_loader)

            print(f'|{epoch_loss:9.4f}|{epoch_acc:8.4f}|{epoch_iou:7.4f}', end='')
            log.write(f'|{epoch_loss:9.4f}|{epoch_acc:8.4f}|{epoch_iou:7.4f}')

            train_loss_record.append(epoch_loss)
            train_acc_record.append(epoch_acc)
            train_iou_record.append(epoch_iou)

            with torch.no_grad():
                epoch_val_loss = 0
                epoch_val_acc = 0
                epoch_val_iou = 0
                for batch_idx, batch in enumerate(val_loader):
                    data, target = batch['image'].to(device), batch['mask'].to(device)

                    output = net(data)
                    loss = net.criterion(output, target.long())
                    acc = accuracy(output, target)
                    iou_ = iou(output, target.int())

                    epoch_val_loss += loss.item() / len(val_loader)
                    epoch_val_acc += acc.item() / len(val_loader)
                    epoch_val_iou += iou_.item() / len(val_loader)

                print(f'|{epoch_val_loss:9.4f}|{epoch_val_acc:8.4f}|{epoch_val_iou:7.4f}|')
                log.write(f'|{epoch_val_loss:9.4f}|{epoch_val_acc:8.4f}|{epoch_val_iou:7.4f}|\n')
                val_loss_record.append(epoch_val_loss)
                val_acc_record.append(epoch_val_acc)
                val_iou_record.append(epoch_val_iou)

                if epoch_val_iou > best_val_iou:
                    best_val_iou = epoch_val_iou
                    torch.save(net.state_dict(), f'{weight_path}_fold{fold}/cycle_{cycle}')

        print(f'Best --> {best_val_iou}')
        log.write(f'Best --> {best_val_iou}\n')
        print('Time used', time.time() - start)
