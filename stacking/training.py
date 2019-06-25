from collections import OrderedDict
import copy

import numpy as np
from tqdm import tqdm

import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from models import CRNN
import utils


def train_and_predict(x_train, y_train, x_val, y_val, x_test):
    """Train a neural network classifier and compute predictions.

    Args:
        x_train (np.ndarray): Training instances.
        y_train (np.ndarray): Training labels.
        x_val (np.ndarray): Validation instances.
        y_val (np.ndarray): Validation labels.
        x_test (np.ndarray): Test instances.

    Returns:
        The predictions of the classifier.
    """
    _ensure_reproducibility()

    # Determine which device (GPU or CPU) to use
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Convert data into PyTorch tensors
    x_train = torch.FloatTensor(x_train).transpose(1, 2)
    x_val = torch.FloatTensor(x_val).transpose(1, 2)
    x_test = torch.FloatTensor(x_test).transpose(1, 2)
    y_train = torch.FloatTensor(y_train)
    y_val = torch.FloatTensor(y_val)

    # Instantiate neural network
    n_classes = y_train.shape[-1]
    n_feats = x_train.shape[1]
    net = CRNN(n_classes, n_feats).to(device)

    # Use binary cross-entropy loss function
    criterion = BCELoss()
    # Use Adam optimization algorithm
    optimizer = Adam(net.parameters(), lr=0.01)
    # Use scheduler to decay learning rate regularly
    scheduler = StepLR(optimizer, step_size=2, gamma=0.9)
    # Use helper class to iterate over data in batches
    loader_train = DataLoader(TensorDataset(x_train, y_train),
                              batch_size=128, shuffle=True)
    loader_val = DataLoader(TensorDataset(x_val, y_val), batch_size=128)

    # Instantiate Logger to record training/validation performance
    # Configure to save the states of the top 3 models during validation
    logger = Logger(net, n_states=3)

    for epoch in range(15):
        # Train model using training set
        pbar = tqdm(loader_train)
        pbar.set_description('Epoch %d' % epoch)
        train(net.train(), criterion, optimizer, pbar, logger, device)

        # Evaluate model using validation set and monitor F1 score
        validate(net.eval(), criterion, loader_val, logger, device)
        logger.monitor('val_f1')

        # Print training and validation results
        logger.print_results()

        # Invoke learning rate scheduler
        scheduler.step()

    # Predict on CPU to save GPU memory
    net.cpu()

    # Ensemble top 3 model predictions
    y_preds = []
    for state_dict in logger.state_dicts:
        net.load_state_dict(state_dict)
        with torch.no_grad():
            y_preds.append(_flatten(net(x_test)).numpy())
    return np.mean(y_preds, axis=0)


def train(net, criterion, optimizer, loader, logger, device=None):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        output = net(batch_x.to(device))
        loss = criterion(output, batch_y.to(device))
        loss.backward()
        optimizer.step()

        logger.log('loss', loss.item())


def validate(net, criterion, loader, logger, device=None):
    for batch_x, batch_y in loader:
        with torch.no_grad():
            output = net(batch_x.to(device))

        loss = criterion(output, batch_y.to(device))
        logger.log('val_loss', loss.item())

        f1_score = utils.f1_score(_flatten(batch_y).cpu(),
                                  _flatten(output).cpu())
        logger.log('val_f1', f1_score)


def _flatten(x):
    # [M, N, K]->[M * N, K]
    return x.view(-1, x.shape[-1]).data


def _ensure_reproducibility():
    np.random.seed(1000)
    torch.manual_seed(1000)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger:
    def __init__(self, net, n_states=0):
        self.net = net
        self.top_k = [0] * n_states
        self.state_dicts = [None] * n_states
        self.results = OrderedDict()

    def log(self, key, value):
        if self.results.get(key) is None:
            self.results[key] = []
        self.results[key].append(value)

    def monitor(self, key):
        value = np.mean(self.results[key])
        if len(self.top_k) > 0 and value > min(self.top_k):
            idx = np.argmin(self.top_k)
            self.top_k[idx] = value
            self.state_dicts[idx] = copy.deepcopy(self.net.state_dict())

    def print_results(self, reset=True):
        print(', '.join(['{}: {}'.format(k, np.mean(v))
                         for k, v in self.results.items()]))

        if reset:
            self.results.clear()
