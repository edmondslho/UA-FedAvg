from InfantMovementCNN import InfantMovementCNN
from flwr.client import ClientApp
from client import FlowerClient

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torch
from util import set_seed

import numpy as np


def make_client_app(fold_id, seed=42, dataSplit="111-211"):
    def client_fn(context):
        cid = int(context.node_config["partition-id"]) + 1  # +1 for filename

        foldID = fold_id

        # start loading data
        # the dataset is collected from Kulvicius et al., Communications medicine, 2025
        x_train_pos = np.loadtxt(
            'data/data_vitpose_3clients_9fold-' + dataSplit + '/client_' + str(
                cid) + '_input_vitpose_9fold_fold_' + str(
                foldID) + '_train.txt')
        x_test_pos = np.loadtxt(
            'data/data_vitpose_3clients_9fold-' + dataSplit + '/client_' + str(
                cid) + '_input_vitpose_9fold_fold_' + str(
                foldID) + '_test.txt')
        y_train = np.loadtxt(
            'data/data_vitpose_3clients_9fold-' + dataSplit + '/client_' + str(
                cid) + '_output_vitpose_9fold_fold_' + str(
                foldID) + '_train.txt')
        y_test = np.loadtxt(
            'data/data_vitpose_3clients_9fold-' + dataSplit + '/client_' + str(
                cid) + '_output_vitpose_9fold_fold_' + str(
                foldID) + '_test.txt')

        x_train_pos = np.reshape(x_train_pos, (x_train_pos.shape[0], 250, 30), order='F')
        x_test_pos = np.reshape(x_test_pos, (x_test_pos.shape[0], 250, 30), order='F')

        x_train = np.zeros((x_train_pos.shape[0], 250, 60))
        x_test = np.zeros((x_test_pos.shape[0], 250, 60))

        x_train[:, :, 0:30] = x_train_pos
        x_train[:, 1:250, 30:60] = np.diff(x_train_pos, axis=1)

        x_test[:, :, 0:30] = x_test_pos
        x_test[:, 1:250, 30:60] = np.diff(x_test_pos, axis=1)

        del x_train_pos
        del x_test_pos

        ### apply z-score normalisation for position and velocity
        x_mean_pos = np.mean(x_train[:, :, 0:30])
        x_sd_pos = np.std(x_train[:, :, 0:30])

        x_mean_vel = np.mean(x_train[:, :, 30:60])
        x_sd_vel = np.std(x_train[:, :, 30:60])

        x_train[:, :, 0:30] = (x_train[:, :, 0:30] - x_mean_pos) / x_sd_pos
        x_train[:, :, 30:60] = (x_train[:, :, 30:60] - x_mean_vel) / x_sd_vel

        x_test[:, :, 0:30] = (x_test[:, :, 0:30] - x_mean_pos) / x_sd_pos
        x_test[:, :, 30:60] = (x_test[:, :, 30:60] - x_mean_vel) / x_sd_vel

        print('x_train size:', x_train.shape)
        print('y_train size:', y_train.shape)
        print('x_test size:', x_test.shape)
        print('y_test size:', y_test.shape)

        x_val = x_train[int(7 * x_train.shape[0] / 8):x_train.shape[0], :, :]
        y_val = y_train[int(7 * y_train.shape[0] / 8):y_train.shape[0]]

        # remove train data
        x_train = x_train[0:int(7 * x_train.shape[0] / 8), :, :]
        y_train = y_train[0:int(7 * y_train.shape[0] / 8)]

        # load data - (N, 250, 60)
        x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
        x_val = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1)
        x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)

        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_val, y_val)
        test_ds = TensorDataset(x_test, y_test)

        # fix the seed
        set_seed(seed)

        # using the CNN model proposed in Kulvicius et al., Communications medicine, 2025
        model = InfantMovementCNN()

        # create data loaders
        trainloader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
        valloader = DataLoader(val_ds, batch_size=4, shuffle=False)
        testloader = DataLoader(test_ds, batch_size=4, shuffle=False)

        return FlowerClient(model, trainloader, valloader, testloader, cid=cid, fid=foldID, dataSplit=dataSplit,
                            seed=seed).to_client()

    return ClientApp(client_fn)
