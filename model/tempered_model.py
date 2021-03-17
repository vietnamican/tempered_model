import torch
from torch import nn
import pytorch_lightning as pl
from typing import Type, Any, Callable, Union, List, Optional

from .base import Base, ConvBatchNormRelu


class TemperedModel(Base):

    def __init__(self, orig, tempered, mode, orig_module_names, tempered_module_names, is_trains, prun_module=None):
        super().__init__()
        self.orig = orig
        self.tempered = tempered
        if prun_module is not None:
            self.prun_module = prun_module
        self._setup_init(mode, orig_module_names,
                         tempered_module_names, is_trains)
        self.logger.experiment.log_text(self.__str__())

    def _log_loss(self, phase, module_names, loss):
        name = phase + '_'
        if isinstance(module_names, list):
            name += "+".join(module_names)
        else:
            name += module_names
        self.log('loss_{}'.format(name), loss)

    def _forward(self, modules, x):
        if isinstance(modules, list):
            inp = x
            for module in modules:
                inp = module(inp)
            out = inp
        else:
            out = modules(x)
        return out

    def forward(self, x, phase='train'):
        if self.mode in ['training', 'tuning', 'inference', 'fast_tuning']:
            x = self.forward_path(x)
            return x
        elif self.mode == 'temper':
            losses = 0.0
            for (
                orig_module_names,
                orig_modules,
                tempered_module_names,
                tempered_modules,
                is_train
            ) in zip(
                self.orig_module_names,
                self.orig_modules,
                self.tempered_module_names,
                self.tempered_modules,
                self.is_trains
            ):
                out = self._forward(orig_modules, x)
                if is_train:
                    _out = self._forward(tempered_modules, x)
                    loss = self.criterion(out, _out)
                    self._log_loss(phase, orig_module_names, loss)
                    losses += loss
                x = out
            return losses
        else:
            print(
                "Not in one of modes: ['training', 'temper', 'tuning', 'inference']")
            return x

    def temper_forward(self, x):
        return self.temper_forward_path(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.mode in ['training', 'tuning', 'fast_tuning']:
            logit = self.forward(x, 'train')
            loss = self.criterion(logit, y)
            pred = logit.argmax(dim=1)
            self.log('train_loss', loss)
            self.log('train_acc_step', self.accuracy(pred, y))
            return loss
        elif self.mode == 'temper':
            loss = self.forward(x, 'train')
            self.log('train_loss', loss)
            return loss

    def training_epoch_end(self, outs):
        if self.mode in ['training', 'tuning', 'fast_tuning']:
            self.log('train_acc_epoch', self.accuracy.compute())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.mode in ['training', 'tuning', 'fast_tuning']:
            logit = self.forward(x, 'val')
            loss = self.criterion(logit, y)
            pred = logit.argmax(dim=1)
            self.log('val_loss', loss)
            self.log('val_acc_step', self.val_accuracy(pred, y))
            return loss
        elif self.mode == 'temper':
            loss = self.forward(x, 'val')
            self.log('val_loss', loss)
            return loss

    def validation_epoch_end(self, outs):
        if self.mode in ['training', 'tuning', 'temper', 'fast_tuning']:
            self.log('val_acc_epoch', self.val_accuracy.compute())

    def _criterion(self, x, y):
        logit = self.temper_forward(x)
        loss = self.classification_criterion(logit, y)
        pred = logit.argmax(dim=1)
        return loss, pred

    def _prun(self, *args, **kwargs):
        self.prun_module.prun(*args, **kwargs)

    def prun(self, *args, **kwargs):
        self._prun(*args, **kwargs)

    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.mode in ['training', 'inference', 'tuning', 'fast_tuning']:
            logit = self.forward(x)
            loss = self.criterion(logit, y)
            pred = logit.argmax(dim=1)
            self.log('test_loss', loss)
            self.log('test_acc_step', self.test_accuracy(pred, y))
            return loss
        elif self.mode == 'temper':
            loss, pred = self._criterion(x, y)
            self.log('test_loss', loss)
            self.log('test_acc_step', self.test_accuracy(pred, y))
            return loss

    def test_epoch_end(self, outputs):
        self.log('test_acc_epoch', self.test_accuracy.compute())

    def configure_optimizers(self):
        print("---------------------------------------------------------")
        print("Load configure from base")
        print("---------------------------------------------------------")
        if self.mode == 'training':
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1,
                                        momentum=0.9, weight_decay=5e-4)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=200)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        elif self.mode == 'temper':
            params = []
            for tempered_modules in self.tempered_modules:
                if isinstance(tempered_modules, list):
                    for tempered_module in tempered_modules:
                        params.extend(tempered_module.parameters())
                else:
                    params.extend(tempered_modules.parameters())
            optimizer = torch.optim.SGD(params, lr=0.01,
                                        momentum=0.9, weight_decay=5e-4)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=200)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        elif self.mode in ['tuning', 'fast_tuning']:
            optimizer = torch.optim.SGD(self.parameters(), lr=0.001,
                                        momentum=0.9, weight_decay=5e-4)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=200)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        else:
            print(
                "Not in one of modes ['training', 'temper', 'tuning', 'fast_tuning']")

    def migrate(self, state_dict, *args, **kwargs):
        state_dict = self.remove_prefix_state_dict(state_dict, 'forward_path')
        super().migrate(state_dict, *args, **kwargs)

    def _setup_init(self, mode, orig_module_names, tempered_module_names, is_trains):
        self.mode = mode
        self.orig_module_names = orig_module_names
        self.tempered_module_names = tempered_module_names
        self.is_trains = is_trains
        self.register_modules()
        self._set_forward_path()
        self._set_criterion()
        self._set_metrics()
        if mode == 'training':
            self.freeze_with_prefix('tempered')
        elif mode == 'temper':
            self.freeze_with_prefix('')
            for module_names, is_train in zip(self.tempered_module_names, self.is_trains):
                if is_train:
                    if isinstance(module_names, list):
                        for module_name in module_names:
                            self.defrost_with_prefix(module_name)
                    else:
                        self.defrost_with_prefix(module_names)
        elif mode in ['inference', 'tuning']:
            for orig_module_names, tempered_module_names, is_train in zip(self.orig_module_names, self.tempered_module_names, self.is_trains):
                if is_train:
                    freeze_modules = orig_module_names
                else:
                    freeze_modules = tempered_module_names
                if isinstance(freeze_modules, list):
                    for freeze_module in freeze_modules:
                        self.freeze_with_prefix(freeze_module)
                else:
                    self.freeze_with_prefix(freeze_modules)
        elif mode == 'fast_tuning':
            self.freeze_with_prefix('')
            for module_names, is_train in zip(self.tempered_module_names, self.is_trains):
                if is_train:
                    if isinstance(module_names, list):
                        for module_name in module_names:
                            self.defrost_with_prefix(module_name)
                    else:
                        self.defrost_with_prefix(module_names)

    def _set_forward_path(self):
        if self.mode == 'training':
            modules = []
            for orig_modules in self.orig_modules:
                if isinstance(orig_modules, list):
                    modules.extend(orig_modules)
                else:
                    modules.append(orig_modules)
            self.forward_path = nn.Sequential(*modules)
        elif self.mode in ['tuning', 'inference', 'fast_tuning']:
            modules = []
            for orig_modules, tempered_modules, is_train in zip(self.orig_modules, self.tempered_modules, self.is_trains):
                if is_train:
                    current_modules = tempered_modules
                else:
                    current_modules = orig_modules
                if isinstance(current_modules, list):
                    modules.extend(current_modules)
                else:
                    modules.append(current_modules)
            self.forward_path = nn.Sequential(*modules)
        elif self.mode == 'temper':
            modules = []
            for orig_modules, tempered_modules, is_train in zip(self.orig_modules, self.tempered_modules, self.is_trains):
                if is_train:
                    current_modules = tempered_modules
                else:
                    current_modules = orig_modules
                if isinstance(current_modules, list):
                    modules.extend(current_modules)
                else:
                    modules.append(current_modules)
            print("set temper forward path")
            self.temper_forward_path = nn.Sequential(*modules)

    def _set_criterion(self):
        criterions = {
            'training': nn.CrossEntropyLoss(),
            'temper': nn.MSELoss(),
            'tuning': nn.CrossEntropyLoss(),
            'inference': nn.CrossEntropyLoss(),
            'fast_tuning': nn.CrossEntropyLoss()
        }
        self.criterion = criterions[self.mode]

    def _set_metrics(self):
        if self.mode in ['training', 'tuning', 'inference', 'fast_tuning']:
            self.accuracy = pl.metrics.Accuracy()
            self.val_accuracy = pl.metrics.Accuracy()
            self.test_accuracy = pl.metrics.Accuracy()

    def register_modules(self):
        self.orig_modules = []
        self.tempered_modules = []
        modules_dict = dict(self.named_modules())
        for module_names in self.orig_module_names:
            if isinstance(module_names, str):
                if module_names in modules_dict:
                    self.orig_modules.append(modules_dict[module_names])
            elif isinstance(module_names, list):
                modules = []
                for module_name in module_names:
                    modules.append(modules_dict[module_name])
                self.orig_modules.append(modules)
        for module_names in self.tempered_module_names:
            if isinstance(module_names, str):
                if module_names in modules_dict:
                    self.tempered_modules.append(modules_dict[module_names])
            elif isinstance(module_names, list):
                modules = []
                for module_name in module_names:
                    modules.append(modules_dict[module_name])
                self.tempered_modules.append(modules)

    def export(self, name, save_weight_only=True):
        if save_weight_only:
            torch.save(self.forward_path.state_dict(), name)
        else:
            torch.save(self.forward_path, name)


class LogitTuneModel(Base):
    def __init__(self, Model, orig_module_names, tempered_module_names, is_trains, device, checkpoint_path=""):
        super().__init__()
        self.reference_model = Model(
            'inference', orig_module_names, tempered_module_names, is_trains=[False]*len(is_trains))
        self.training_model = Model(
            'inference', orig_module_names, tempered_module_names, is_trains)

        if checkpoint_path is not None and len(checkpoint_path) > 0:
            if device == 'cpu' or device == 'tpu':
                checkpoint = torch.load(
                    checkpoint_path, map_location=lambda storage, loc: storage)
            else:
                checkpoint = torch.load(checkpoint_path)
            state_dict = checkpoint['state_dict']
            self.reference_model.migrate(state_dict)
            self.training_model.migrate(state_dict)
        self.criterion = nn.MSELoss()
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        reference_logit = self.reference_model(x)
        training_logit = self.training_model(x)

        return reference_logit, training_logit

    def training_step(self, batch, batch_idx):
        x, y = batch
        reference_logit, training_logit = self.forward(x)
        loss = self.criterion(training_logit, reference_logit)
        self.log('train_loss', loss)
        reference_pred = reference_logit.argmax(dim=1)
        training_pred = training_logit.argmax(dim=1)
        self.log('train_acc_step', self.train_accuracy(
            reference_pred, training_pred))
        return loss

    def training_epoch_end(self, outputs):
        self.log('train_acc_epoch', self.train_accuracy.compute())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        reference_logit, training_logit = self.forward(x)
        loss = self.criterion(training_logit, reference_logit)
        self.log('val_loss', loss)
        reference_pred = reference_logit.argmax(dim=1)
        training_pred = training_logit.argmax(dim=1)
        self.log('val_acc_step', self.val_accuracy(
            reference_pred, training_pred))
        return loss

    def validation_epoch_end(self, outs):
        self.log('val_acc_epoch', self.val_accuracy.compute())

    def test_step(self, batch, batch_idx):
        x, y = batch
        reference_logit, training_logit = self.forward(x)
        loss = self.criterion(training_logit, reference_logit)
        self.log('test_loss', loss)
        reference_pred = reference_logit.argmax(dim=1)
        training_pred = training_logit.argmax(dim=1)
        self.log('test_acc_step', self.test_accuracy(
            reference_pred, training_pred))
        return loss

    def test_epoch_end(self, output):
        self.log('test_acc_epoch', self.test_accuracy.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.training_model.parameters(), lr=0.001,
                                    momentum=0.9, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
