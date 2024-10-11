import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn

from config import *
from dataset import get_data_loaders, load_dataset
from model import LSTM_PM
from utils import *


def get_model(temporal, device):
    model = LSTM_PM(temporal = temporal)
    model = model.to(device)
    return model


def loss_function(pred_maps, gt_maps, temporal):
    criterion = nn.MSELoss(reduction = 'mean')
    initial_hitmaps = pred_maps[0]
    gt = gt_maps[:, 0, :, :, :]
    initial_loss = criterion(initial_hitmaps, gt)
    total_loss = initial_loss

    for t in range(temporal):
        pred = pred_maps[t + 1]
        gt = gt_maps[:, t, :, :, :]
        # Loss of each stage
        s_loss = criterion(pred, gt)
        total_loss += s_loss

    return total_loss


def train(model, dataloader, optimizer, criterion, epoch, temporal, device):
    # Put the model on train mode
    model.train()
    losses = []
    total_predictions = get_predictions_dict()

    for iteration, (images, gt_maps, center_map, maxbbox_list, imgs) in enumerate(dataloader):

        images = images.to(device)
        gt_maps = gt_maps.to(device)
        center_map = center_map.to(device)

        pred_heatmaps = model(images, center_map)

        loss = criterion(pred_heatmaps, gt_maps, temporal)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = compute_metric(pred_heatmaps, gt_maps.cpu().numpy(),
                                     maxbbox_list.numpy(), temporal)
        # Update total predictions
        for dict in predictions:
            for key in predictions[dict]:
                total_predictions[dict][key] += predictions[dict][key]

        if iteration % round((len(dataloader) / 5)) == 0:
            acc = total_predictions['total']['correct'] / total_predictions['total']['all']
            print(
                f'/r[Epoch][Batch] = [{epoch + 1}][{iteration}] -> Loss = {np.mean(losses):.2f} | PCK accuracy = {acc:.2f}')

    return np.mean(losses), total_predictions


def evaluate(model, dataloader, criterion, temporal, device):
    # Put the model on evaluation mode
    model.eval()
    losses = []
    total_predictions = get_predictions_dict()

    for iteration, (images, gt_maps, center_map, maxbbox_list, imgs) in enumerate(dataloader):

        images = images.to(device)
        gt_maps = gt_maps.to(device)
        center_map = center_map.to(device)

        pred_heatmaps = model(images, center_map)

        loss = criterion(pred_heatmaps, gt_maps, temporal)
        losses.append(loss.item())

        predictions = compute_metric(pred_heatmaps,
                                     gt_maps.cpu().numpy(),
                                     maxbbox_list.numpy(),
                                     temporal)
        # update total predictions
        for dict in predictions:
            for key in predictions[dict]:
                total_predictions[dict][key] += predictions[dict][key]

    return np.mean(losses), total_predictions


def trainModel(dataset_path, temporal, lr, train_bs, eval_bs, epocks, weight_decay, sch_gamma, sch_step, title=''):
    # Loading dataset
    train_videos, train_labels, test_videos, test_labels = load_dataset(dataset_path)

    # Instantiate data loaders
    train_dataloader, val_dataloader = get_data_loaders(train_videos, train_labels, test_videos,
                                                        test_labels, train_bs, eval_bs)

    print('-' * 40)
    print('Number of train batches =', len(train_dataloader))
    print('Number of validation batches =', len(val_dataloader))

    print('-' * 40)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, 'is available')

    # Initiating the neural network
    model = get_model(temporal, device)

    # Determining the type of optimizer, scheduling and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-08, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    criterion = loss_function

    best_accuracy = 0
    loss_list, metric_list = [], []
    print('Start Training ....', end='')
    for epock in range(epocks):
        train_loss, train_predictions = train(model, train_dataloader, optimizer, criterion, epock, temporal, device)
        val_loss, val_predictions = evaluate(model, val_dataloader, criterion, temporal, device)
        scheduler.step()

        train_acc = get_acc(train_predictions)
        val_acc = get_acc(val_predictions)

        loss_list.append([train_loss, val_loss])
        metric_list.append(val_acc)

        t_a, v_a = train_acc['total'], val_acc['total']
        print(f'\tTrain -> Loss = {train_loss:.4f} / PCK accuracy = {t_a:.4f}')
        print(f'\tValidation -> Loss = {val_loss:.4f} / PCK accuracy = {v_a:.4f}')

    plot(np.array(loss_list), np.array(metric_list), title)

    # Lưu mô hình sau khi huấn luyện
    torch.save(model.state_dict(), 'model_weights.pth')
    print('Model weights saved to model_weights.pth')

    return model, loss_list, metric_list


if __name__ == "__main__":
    model, loss_list, metric_list = trainModel(DATASET_PATH,
                                               TEMPORAL, LR,
                                               TRAIN_BS,
                                               EVAL_BS,
                                               EPOCHS,
                                               WEIGHT_DECAY,
                                               SCH_GAMMA,
                                               SCH_STEP)
