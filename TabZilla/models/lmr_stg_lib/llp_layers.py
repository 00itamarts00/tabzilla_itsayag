import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .meter import GroupMeters
from torch.distributions.uniform import Uniform
from torch.nn import init
from torch.utils.data import DataLoader

from .utils import (FastTensorDataLoader, SimpleDataset, as_cpu, as_float,
                   as_numpy, as_tensor, get_activation, get_optimizer,
                   probe_infnan)

# Yutaro's implemention


class LinearLayer(nn.Sequential):
    def __init__(self, in_features, out_features, batch_norm=None, dropout=None, bias=None, activation=None):
        if bias is None:
            bias = (batch_norm is None)

        modules = [nn.Linear(in_features, out_features, bias=bias)]
        if batch_norm is not None and batch_norm is not False:
            modules.append(get_batcnnorm(batch_norm, out_features, 1))
        if dropout is not None and dropout is not False:
            modules.append(get_dropout(dropout, 1))
        if activation is not None and activation is not False:
            modules.append(get_activation(activation))
        super().__init__(*modules)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, batch_norm=None, dropout=None, activation='relu', flatten=True):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = []
        elif type(hidden_dims) is int:
            hidden_dims = [hidden_dims]

        dims = [input_dim]
        dims.extend(hidden_dims)
        dims.append(output_dim)
        modules = []

        nr_hiddens = len(hidden_dims)
        for i in range(nr_hiddens):
            layer = LinearLayer(
                dims[i], dims[i+1], batch_norm=batch_norm, dropout=dropout, activation=activation)
            modules.append(layer)
        layer = nn.Linear(dims[-2], dims[-1], bias=True)
        modules.append(layer)
        self.mlp = nn.Sequential(*modules)
        self.flatten = flatten

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, input):
        if self.flatten:
            input = input.view(input.size(0), -1)
        return self.mlp(input)


class ModelIOKeysMixin(object):
    def _get_input(self, feed_dict):
        return feed_dict['input']

    def _get_label(self, feed_dict):
        return feed_dict['label']

    def _get_covariate(self, feed_dict):
        '''For cox'''
        return feed_dict['X']

    def _get_fail_indicator(self, feed_dict):
        '''For cox'''
        return feed_dict['E'].reshape(-1, 1)

    def _get_failure_time(self, feed_dict):
        '''For cox'''
        return feed_dict['T']

    def _compose_output(self, value):
        return dict(pred=value)

# Ofir's implementation


class ConcreteLayer(nn.Module):
    def __init__(self, input_dim, output_dim, start_temp, min_temp, nr_epochs):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nr_epochs = nr_epochs
        self.start_temp = start_temp
        self.min_temp = torch.tensor(min_temp)
        # self.logits = torch.nn.Parameter(torch.zeros(self.input_dim, self.output_dim), requires_grad=True)

    def current_temp(self, epoch, sched_type='exponential'):
        schedules = {
            'exponential': torch.max(self.min_temp, self.start_temp * ((self.min_temp / self.start_temp) ** (epoch / self.nr_epochs))),
            'linear': torch.max(self.min_temp, self.start_temp - (self.start_temp - self.min_temp) * (epoch / self.nr_epochs)),
            'cosine': self.min_temp + 0.5 * (self.start_temp - self.min_temp) * (1. + np.cos(epoch * math.pi / self.nr_epochs))
        }
        return schedules[sched_type]

    def forward(self, x, logits, epoch=None):
        self.logits = logits
        uniform_pdfs = Uniform(low=1e-6, high=1.).sample(self.logits.size()).to(x.device)
        gumbel = -torch.log(-torch.log(uniform_pdfs))

        if self.training:
            temp = self.current_temp(epoch)
            noisy_logits = (self.logits + gumbel) / temp
            weights = F.softmax(noisy_logits, dim=1)  # Note: dim is 1
            x = torch.diagonal(x @ weights.T).reshape(-1, 1)
        else:
            weights = F.one_hot(torch.argmax(
                self.logits, dim=1), self.input_dim).float()
            x = torch.diagonal(x @ weights.T).reshape(-1, 1)
        return x, weights

    # Deprecated: this only returns the weights for the last batch
    def get_weights(self, epoch):
        temp = self.current_temp(epoch)
        return F.softmax(self.logits / temp, dim=0)
    # Deprecated: this only returns the feats for the last batch

    def get_selected_feats(self):
        feats = torch.argmax(self.logits, dim=0)
        return feats


class LLPmodel(MLPLayer, ModelIOKeysMixin):
    def __init__(self, device, input_dim, hidden_dims, k,  start_temp, min_temp, batch_norm=None, dropout=None, activation='relu', nr_epochs=1000):
        super().__init__(input_dim, k, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)

        # parameter registration
        self.device = device
        self.k = k
        self.loss = nn.MSELoss()  # TODO: add options for other losses

        self.thetas = torch.nn.Parameter(init.uniform_(torch.zeros(
            input_dim, self.k)), requires_grad=True)  # Q: what is the best way to initialize?
        self.ConcreteLayer = ConcreteLayer(
            k, 1, start_temp, min_temp, nr_epochs)

    def forward(self, feed_dict, epoch):
        x = self._get_input(feed_dict)
        # MLP layer to parameterize alpha
        # logits shape: n * k Q: should I put the structure as it is or should I instantiate a MLP inside the class instead?
        self.logits = super().forward(x).reshape(-1, self.k)
        selected_predictor, weights = self.ConcreteLayer(
            x@self.thetas, self.logits, epoch)  # X@self.thetas shape: n * k

        if self.training:
            # loss = self.loss(logits, self._get_fail_indicator(feed_dict), self.noties)
            loss = self.loss(selected_predictor, self._get_label(feed_dict))
            total_loss = loss
            return total_loss
        else:
            return self._compose_output(selected_predictor)

    def get_thetas(self):
        return self.thetas.detach().cpu().numpy()

    # Note: make sure self.training is False
    def get_logits_torch(self, X):
        self.ConcreteLayer.eval()
        with torch.no_grad():
            return super().forward(X).reshape(-1, self.k)

    def get_logits(self, X):
        self.ConcreteLayer.eval()
        with torch.no_grad():
            return super().forward(X).reshape(-1, self.k).detach().cpu().numpy()

    def get_weights(self, X, epoch):
        self.ConcreteLayer.eval()
        with torch.no_grad():
            selected_predictor, weights = self.ConcreteLayer(
                X@self.thetas, self.get_logits_torch(X), epoch)
        return selected_predictor, weights


class LLP(object):
    """Local Linear Predictor:

    Parameters
    ----------
    device : str, default "cuda"
        device types, "cuda" or "cpu"
    input_dim : int, default 100
        Dimension of input dataset (total #features)
    hidden_dims : list, default [30]
        Number of nodes in each hidden layer of MLP that outputs logits
    k : int, default 3
        Number of predictors
    start_temp : float, default 10
        Initial temperature in the concrete layer
    min_temp : float, default 1e-2
        Final temperature in the concrete layer
    optimizer: str or optim.Optimizer, default "Adam"
        Optimizer setting
    learning_rate: float, default 1e-5
        learning rate of the optimizer
    weight_decay: float, default 1e-3
        weight decay of the optimizer
    batch_norm: bool, default None
        whether do batch norm
    dropout: bool, default None
        whether dropout
    activation: str, default "relu"
        activation function for MLP
    """

    def __init__(self, device="cuda", input_dim=100, hidden_dims=[30], k=3, start_temp=10, min_temp=1e-2, optimizer="Adam", learning_rate=1e-5, weight_decay=1e-3, batch_size=100, batch_norm=None, dropout=None, activation='relu', nr_epochs=1000):
        # parameter registration
        self.device = device
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.k = k
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.activation = activation
        self.nr_epochs = nr_epochs
        self.metric = nn.MSELoss()  # TODO: add options for other losses
        # initialize a LLPmodel and the optimizer
        self._model = LLPmodel(device, input_dim, hidden_dims, k, start_temp,
                               min_temp, batch_norm, dropout, activation, nr_epochs)
        self._model.apply(self.init_weights)
        self._model = self._model.to(self.device)
        self._optimizer = get_optimizer(
            optimizer, self._model, lr=learning_rate, weight_decay=weight_decay)

    # xavier_uniform initialization
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    # overall wrapper
    def fit(self, X, y, valid_X=None, valid_y=None,
            verbose=True, print_interval=1, shuffle=False, meters=None):
        data_loader = self.get_dataloader(X, y, shuffle)

        if valid_X is not None:
            val_data_loader = self.get_dataloader(valid_X, valid_y, shuffle)
        else:
            val_data_loader = None
        self.train(data_loader, val_data_loader,
                   verbose, print_interval, meters)

    # loop over epochs
    def train(self, data_loader, val_data_loader=None, verbose=True, print_interval=1, meters=None):
        if meters is None:
            meters = GroupMeters()
        for epoch in range(1, 1 + self.nr_epochs):
            self.train_epoch(data_loader, epoch, meters)
            if verbose and epoch % print_interval == 0:
                self.validate(val_data_loader, self.metric, epoch)
                caption = 'Epoch: {}:'.format(epoch)
                print(meters.format_simple(caption))

    # loop over batches
    def train_epoch(self, data_loader, epoch, meters=None):
        if meters is None:
            meters = GroupMeters()
        self._model.train()
        for feed_dict in data_loader:
            self.train_step(feed_dict, epoch, meters)

    # per batch training
    def train_step(self, feed_dict, epoch, meters=None):
        assert self._model.training

        loss = self._model(feed_dict, epoch)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        loss = as_float(loss)

        if meters is not None:
            meters.update(loss=loss)

    def validate_step(self, feed_dict, metric, epoch, mode='valid', meters=None):
        with torch.no_grad():
            pred = self._model(feed_dict, epoch)
        result = metric(pred['pred'], self._model._get_label(feed_dict))
        if meters is not None:
            meters.update({mode+'_loss': result})

    def validate(self, data_loader, metric, epoch, mode='valid', meters=None):
        if meters is None:
            meters = GroupMeters()
        self._model.eval()
        for fd in data_loader:
            self.validate_step(fd, metric, epoch, mode=mode, meters=meters)

    def predict(self, X, epoch):

        dataset = SimpleDataset(torch.from_numpy(X).float().to(self.device))
        data_loader = DataLoader(dataset, batch_size=X.shape[0], shuffle=False)
        res = []
        self._model.eval()
        for feed_dict in data_loader:
            feed_dict_np = as_numpy(feed_dict)
            feed_dict = as_tensor(feed_dict)
            with torch.no_grad():
                output_dict = self._model(feed_dict, epoch)
            output_dict_np = as_numpy(output_dict)
            res.append(output_dict_np['pred'])
        return np.concatenate(res, axis=0)

    def evaluate(self, X, y, epoch):
        data_loader = self.get_dataloader(X, y, shuffle=None)
        meters = GroupMeters()
        self.validate(data_loader, self.metric, epoch, mode='test')
        print(meters.format_simple(''))

    def save_checkpoint(self, filename, extra=None):
        model = self._model

        state = {
            'model': state_dict(model, cpu=True),
            'optimizer': as_cpu(self._optimizer.state_dict()),
            'extra': extra
        }
        try:
            torch.save(state, filename)
            logger.info('Checkpoint saved: "{}".'.format(filename))
        except Exception:
            logger.exception(
                'Error occurred when dump checkpoint "{}".'.format(filename))

    def load_checkpoint(self, filename):
        if osp.isfile(filename):
            model = self._model
            if isinstance(model, nn.DataParallel):
                model = model.module

            try:
                checkpoint = torch.load(filename)
                load_state_dict(model, checkpoint['model'])
                self._optimizer.load_state_dict(checkpoint['optimizer'])
                logger.critical('Checkpoint loaded: {}.'.format(filename))
                return checkpoint['extra']
            except Exception:
                logger.exception(
                    'Error occurred when load checkpoint "{}".'.format(filename))
        else:
            logger.warning(
                'No checkpoint found at specified position: "{}".'.format(filename))
        return None

    def get_dataloader(self, X, y, shuffle):
        return FastTensorDataLoader(torch.from_numpy(X).float().to(self.device),
                                    torch.from_numpy(y).float().to(self.device), tensor_names=('input', 'label'),
                                    batch_size=self.batch_size, shuffle=shuffle)

    def get_thetas(self):
        self._model.eval()
        with torch.no_grad():
            trained_thetas = self._model.get_thetas()
        return trained_thetas

    # the model selection prob for each sample
    def get_weights(self, X, epoch):
        self._model.eval()
        with torch.no_grad():
            selected_predictor, weights = self._model.get_weights(
                torch.from_numpy(X).float().to(self.device), epoch)
        return selected_predictor, weights
