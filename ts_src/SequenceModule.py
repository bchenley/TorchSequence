import pytorch_lightning as pl
import torch

import numpy as np
import pandas as pd

import time as run_time

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ts_src.Criterion import Criterion

import pytorch_lightning as pl

class SequenceModule(pl.LightningModule):
  def __init__(self,
               model,
               opt, loss_fn, metric_fn=None,
               constrain=False, penalize=False,
               shuffle_train=False,
               teach=False,
               stateful = False,
               track_performance=False, track_params=False,
               model_dir=None):
      """
      A PyTorch Lightning module for sequence forecasting.

      Args:
          model (torch.nn.Module): The model used for sequence forecasting.
          opt (torch.optim.Optimizer): The optimizer to use during training.
          loss_fn (callable): The loss function to calculate loss.
          metric_fn (callable, optional): The metric function to track performance. Default is None.
          constrain (bool, optional): Whether to constrain the model's parameters. Default is False.
          penalize (bool, optional): Whether to apply penalty to the model's parameters. Default is False.
          teach (bool, optional): Whether to use teacher forcing during training. Default is False.
          track_performance (bool, optional): Whether to track performance metrics. Default is False.
          track_params (bool, optional): Whether to track model parameters. Default is False.
          model_dir (str, optional): Directory to save model checkpoints. Default is None.
      """
      super().__init__()

      # Set automatic optimization to False
      self.automatic_optimization = False

      self.model = model

      # Set the accelerator based on the device (GPU or CPU)
      self.accelerator = 'gpu' if self.model.device == 'cuda' else 'cpu'

      if isinstance(loss_fn, list):
        if len(loss_fn) == 1:
          loss_fn = loss_fn * model.num_outputs

        metric_fn = [metric_fn] if not isinstance(metric_fn, list) else metric_fn
        if len(metric_fn) == 1:
          metric_fn = metric_fn * model.num_outputs

        opt = [opt] if not isinstance(opt, list) else opt
        if len(opt) == 1:
          opt = opt * model.num_outputs

      self.opt, self.loss_fn, self.metric_fn = opt, loss_fn, metric_fn

      self.constrain, self.penalize = constrain, penalize

      self.teach = teach
      self.stateful = stateful
      self.hiddens = None

      input_size, output_size = self.model.input_size, self.model.output_size

      self.train_history, self.val_history = None, None
      self.current_val_epoch = 0

      self.train_step_loss, self.train_step_metric = [], []
      self.val_step_loss, self.val_step_metric = [], []
      self.test_step_loss, self.test_step_metric = [], []

      self.track_performance, self.track_params = track_performance, track_params

      self.model_dir = model_dir

  def forward(self,
                input,
                hiddens=None,
                steps=None,
                target=None,
                input_window_idx=None, output_window_idx=None,
                output_mask=None,
                output_input_idx=None, input_output_idx=None,
                encoder_output=None):
      """
      Forward pass through the model.

      Args:
          input (torch.Tensor): Input tensor for the model.
          hiddens (torch.Tensor, optional): Initial hidden state of the model. Default is None.
          steps (torch.Tensor, optional): Steps for forecasting. Default is None.
          target (torch.Tensor, optional): Target tensor for the model. Default is None.
          input_window_idx (torch.Tensor, optional): Input window indices. Default is None.
          output_window_idx (torch.Tensor, optional): Output window indices. Default is None.
          output_mask (torch.Tensor, optional): Output mask. Default is None.
          output_input_idx (torch.Tensor, optional): Output input indices. Default is None.
          input_output_idx (torch.Tensor, optional): Input output indices. Default is None.
          encoder_output (torch.Tensor, optional): Output from the encoder. Default is None.

      Returns:
          output (torch.Tensor): Output tensor from the model.
          hiddens (torch.Tensor): Updated hidden state of the model.
      """
      # Forward pass through the model
      output, hiddens = self.model.forward(input=input,
                                           steps=steps,
                                           hiddens=hiddens,
                                           target=target,
                                           input_window_idx=input_window_idx,
                                           output_window_idx=output_window_idx,
                                           output_mask=output_mask,
                                           output_input_idx=output_input_idx,
                                           input_output_idx=input_output_idx,
                                           encoder_output=encoder_output)

      return output, hiddens

  ## Configure optimizers
  def configure_optimizers(self):
    """
    Configure and return the optimizer(s) for the model.

    Returns:
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    """
    return self.opt
  ##

  ## train model
  def on_train_start(self):
    """
    Callback function executed at the beginning of the training loop.

    This function records the start time of the training loop.

    Returns:
        None
    """
    self.run_time = run_time.time()

  def training_step(self, batch, batch_idx):
    """
    LightningModule training step.

    Args:
        batch: The input batch from the dataloader.
        batch_idx: Index of the current batch.

    Returns:
        dict: Dictionary containing the loss and metric values for the current batch.
    """

    # Unpack batch
    input_batch, target_batch, steps_batch, batch_size, id = batch

    # Keep the first `batch_size` batches of hiddens
    if not self.stateful:
      self.hiddens = None
    elif self.hiddens is not None:
      for i in range(self.model.num_inputs):
        if (self.model.base_type[i] in ['gru', 'lstm', 'lru']) & (self.hiddens[i] is not None):
          if self.model.base_type[i] == 'lstm':
            if self.hiddens[i][0].shape[1] >= batch_size:
              self.hiddens[i] = tuple([s[:, :batch_size].contiguous() for s in self.hiddens[i]])
            else:
              self.hiddens[i] = tuple([torch.nn.functional.pad(s.contiguous(), pad=(0, 0, 0, batch_size-s.shape[1]), mode='constant', value=0) for s in self.hiddens[i]])
          else:
              if self.hiddens[i].shape[1] >= batch_size:
                self.hiddens[i] = self.hiddens[i][:, :batch_size].contiguous()
              else:
                self.hiddens[i] = torch.nn.functional.pad(self.hiddens[i].contiguous(), pad=(0, 0, 0, batch_size-self.hiddens[i].shape[1]), mode='constant', value=0)

    input_batch = input_batch[:batch_size]
    target_batch = target_batch[:batch_size]
    steps_batch = steps_batch[:batch_size]
    
    # Perform forward pass to compute gradients
    prediction_batch, self.hiddens = self.forward(input=input_batch,
                                                   steps=steps_batch,
                                                   hiddens=self.hiddens,
                                                   target=target_batch if self.teach else None,
                                                   input_window_idx=self.trainer.datamodule.train_input_window_idx,
                                                   output_window_idx=self.trainer.datamodule.train_output_window_idx,
                                                   output_input_idx=self.trainer.datamodule.output_input_idx,
                                                   input_output_idx=self.trainer.datamodule.input_output_idx,
                                                   output_mask=self.trainer.datamodule.train_output_mask)

    # Add penalty loss if desired
    penalty = 0
    if self.penalize: penalty = self.model.penalize()

    prediction_batch_masked = prediction_batch * self.trainer.datamodule.train_output_mask
    target_batch_masked = target_batch * self.trainer.datamodule.train_output_mask

    loss, metric = [], []
    j = 0
    for i in range(self.model.num_outputs):
      if isinstance(self.loss_fn, list):
        opt_i = self.opt[i]
        loss_fn_i = self.loss_fn[i]
        metric_fn_i = self.metric_fn[i]
      else:
        opt_i = self.opt
        loss_fn_i = self.loss_fn
        metric_fn_i = self.metric_fn

      loss_i = sum(loss_fn_i(prediction_batch_masked[...,j:(j+self.model.output_size[i])],
                         target_batch_masked[...,j:(j+self.model.output_size[i])]))
      loss.append(loss_i.unsqueeze(0))

      if metric_fn_i is not None:
        metric_i = sum(metric_fn_i(prediction_batch_masked[...,j:(j+self.model.output_size[i])],
                               target_batch_masked[...,j:(j+self.model.output_size[i])]))
        metric.append(metric_i.unsqueeze(0))

      j += self.model.output_size[i]

    if isinstance(self.loss_fn, list):
      for i in range(len(self.loss_fn)):
        self.opt[i].zero_grad()        
        sum(loss[i]).backward(retain_graph = True)
        self.opt[i].step()
    else:
      self.opt.zero_grad()
      sum(loss).backward()
      self.opt.step()

    loss = torch.cat(loss)
    if metric is not None: metric = torch.cat(metric)

    # Store loss to be used later in `on_train_epoch_end`
    self.train_step_loss.append(loss)

    if metric is not None: self.train_step_metric.append(metric)

    return {"loss": loss, "metric": metric}

  def on_train_batch_start(self, batch, batch_idx):
    """
    LightningModule method called at the start of each training batch.

    Args:
      batch: The input batch from the dataloader.
      batch_idx: Index of the current batch.
    """

    if self.hiddens is not None:
      for i in range(self.model.num_inputs):
        if (self.model.base_type[i] in ['gru', 'lstm', 'lru']) & (self.hiddens[i] is not None):
          if self.model.base_type[i] == 'lstm':
            # Detach hidden states for LSTM
            self.hiddens[i] = [s.detach() for s in self.hiddens[i]]
          else:
            # Detach hidden states for other RNN types
            self.hiddens[i] = self.hiddens[i].detach()

  def on_train_batch_end(self, outputs, batch, batch_idx):

    """
    LightningModule method called at the end of each training batch.

    Args:
        outputs: Dictionary containing the output values from the training step.
        batch: The input batch from the dataloader.
        batch_idx: Index of the current batch.
    """

    # Constrain model if desired
    if self.constrain:
      with torch.no_grad():
        self.model.constrain()

    # Get the loss and metric values of the current batch
    train_step_loss = outputs['loss'].detach()
    train_step_metric = outputs['metric'].detach() if outputs['metric'] is not None else None
    #

    output_names = self.trainer.datamodule.output_names

    # Log and display the sum of batch loss
    for i,loss_value in enumerate(train_step_loss):
      if isinstance(self.loss_fn, list):
        loss_name_i = self.loss_fn[i].name + '_' + output_names[i]
      else:
        loss_name_i = self.loss_fn.name + '_' + output_names[i]

      self.log(f"train_step_{loss_name_i}", train_step_loss[i], on_step=True, prog_bar=True)
    self.log(f"train_step_loss", sum(train_step_loss), on_step=True, prog_bar=False)
    #

    if self.track_performance or self.track_params:
      if self.train_history is None:
        self.current_train_step = 0
        self.train_history = {'step': torch.empty((0, 1)).to(device=train_step_loss.device, dtype=torch.long)}
        if self.track_performance:
            for i in range(self.model.num_outputs):
                metric_name_i = None
                if isinstance(self.loss_fn, list):
                  loss_name_i = self.loss_fn[i].name + '_' + output_names[i]
                  metric_name_i = self.metric_fn[i].name + '_' + output_names[i] if self.metric_fn[i] is not None else None
                else:
                  loss_name_i = self.loss_fn.name + '_' + output_names[i]
                  metric_name_i = self.metric_fn.name + '_' + output_names[i] if self.metric_fn is not None else None

                self.train_history[loss_name_i] = torch.empty((0, 1)).to(train_step_loss)

                if train_step_metric is not None:
                    self.train_history[metric_name_i] = torch.empty((0, 1)).to(train_step_metric)

        if self.track_params:
            for name, param in self.model.named_parameters():
                if param.requires_grad == True:
                    self.train_history[name] = torch.empty((0, param.numel())).to(param)
      else:
        self.train_history['step'] = torch.cat((self.train_history['step'],
                                                  torch.tensor(self.current_train_step).reshape(1, 1).to(train_step_loss)), 0)

        if self.track_performance:
            for i in range(self.trainer.datamodule.num_outputs):
              metric_name_i = None
              if isinstance(self.loss_fn, list):
                loss_name_i = self.loss_fn[i].name + '_' + output_names[i]
                metric_name_i = self.metric_fn[i].name + '_' + output_names[i] if self.metric_fn[i] is not None else None
              else:
                loss_name_i = self.loss_fn.name + '_' + output_names[i]
                metric_name_i = self.metric_fn.name + '_' + output_names[i] if self.metric_fn is not None else None

              self.train_history[loss_name_i] = torch.cat((self.train_history[loss_name_i],
                                                           train_step_loss[i].reshape(1, 1).to(train_step_loss)), 0)

              if train_step_metric is not None:
                self.train_history[metric_name_i] = torch.cat((self.train_history[metric_name_i],
                                                              train_step_metric[i].reshape(1, 1).to(train_step_metric)), 0)

        if self.track_params:
          for i, (name, param) in enumerate(self.model.named_parameters()):
            if param.requires_grad:
              self.train_history[name] = torch.cat((self.train_history[name],
                                                    param.detach().cpu().reshape(1, -1).to(param)), 0)

        self.current_train_step += 1

  def on_train_epoch_start(self):
      """
      LightningModule method called at the start of each training epoch.
      """
      # Reset the hiddens and train_step_loss
      self.train_step_loss = []

  def on_train_epoch_end(self):
    """
    LightningModule method called at the end of each training epoch.
    """
    # Calculate the mean of train step loss across batches
    train_epoch_loss = torch.stack(self.train_step_loss).mean(0)

    output_names = self.trainer.datamodule.output_names

    # Log and display the mean of batch loss
    for i,loss_value in enumerate(train_epoch_loss):
      if isinstance(self.loss_fn, list):
        loss_name_i = self.loss_fn[i].name + '_' + output_names[i]
      else:
        loss_name_i = self.loss_fn.name + '_' + output_names[i]

      self.log(f"train_epoch_{loss_name_i}", train_epoch_loss[i], on_epoch=True, prog_bar=True)
    self.log(f"train_epoch_loss", train_epoch_loss.mean(), on_epoch=True, prog_bar=False)
    #

    # Clear the list of train step loss for the next epoch
    self.train_step_loss.clear()
  ## End of Training

  ## Validate Model
  def validation_step(self, batch, batch_idx):
    """
    LightningModule method called for each validation batch.
    """
    # Unpack the validation batch
    input_batch, target_batch, steps_batch, batch_size, id = batch

    # Keep the first `batch_size` batches of hiddens
    if not self.stateful:
      self.hiddens = None
    elif self.hiddens is not None:
      for i in range(self.model.num_inputs):
        if (self.model.base_type[i] in ['gru', 'lstm', 'lru']) & (self.hiddens[i] is not None):
          if self.model.base_type[i] == 'lstm':
            if self.hiddens[i][0].shape[1] >= batch_size:
              self.hiddens[i] = tuple([s[:, :batch_size].contiguous() for s in self.hiddens[i]])
            else:
              self.hiddens[i] = tuple([torch.nn.functional.pad(s.contiguous(), pad=(0, 0, 0, batch_size-s.shape[1]), mode='constant', value=0) for s in self.hiddens[i]])
          else:
            if self.hiddens[i].shape[1] >= batch_size:
              self.hiddens[i] = self.hiddens[i][:, :batch_size].contiguous()
            else:
              self.hiddens[i] = torch.nn.functional.pad(self.hiddens[i].contiguous(), pad=(0, 0, 0, batch_size-self.hiddens[i].shape[1]), mode='constant', value=0)

    # Slice the batch inputs, outputs, and steps
    input_batch = input_batch[:batch_size]
    target_batch = target_batch[:batch_size]
    steps_batch = steps_batch[:batch_size]

    # Perform forward pass to compute predictions
    prediction_batch, self.hiddens = self.forward(input=input_batch,
                                                   steps=steps_batch,
                                                   hiddens=self.hiddens,
                                                   target=None,
                                                   input_window_idx=self.trainer.datamodule.val_input_window_idx,
                                                   output_window_idx=self.trainer.datamodule.val_output_window_idx,
                                                   output_input_idx=self.trainer.datamodule.output_input_idx,
                                                   input_output_idx=self.trainer.datamodule.input_output_idx,
                                                   output_mask=self.trainer.datamodule.val_output_mask)

    prediction_batch_masked = prediction_batch * self.trainer.datamodule.val_output_mask
    target_batch_masked = target_batch * self.trainer.datamodule.val_output_mask

    metric = None
    loss, metric = [], []
    j = 0
    for i in range(self.model.num_outputs):
      if isinstance(self.loss_fn, list):
        loss_fn_i = self.loss_fn[i]
        metric_fn_i = self.metric_fn[i]
      else:
        loss_fn_i = self.loss_fn
        metric_fn_i = self.metric_fn

      loss_i = sum(loss_fn_i(prediction_batch_masked[...,j:(j+self.model.output_size[i])],
                         target_batch_masked[...,j:(j+self.model.output_size[i])]))

      loss.append(loss_i.unsqueeze(0))

      if metric_fn_i is not None:
        metric_i = sum(metric_fn_i(prediction_batch_masked[...,j:(j+self.model.output_size[i])],
                               target_batch_masked[...,j:(j+self.model.output_size[i])]))
        metric.append(metric_i.unsqueeze(0))

      j += self.model.output_size[i]

    loss = torch.cat(loss)
    if metric is not None: metric = torch.cat(metric)

    # Store loss to be used later in `on_validation_epoch_end`
    self.val_step_loss.append(loss)

    if metric is not None: self.val_step_metric.append(metric)

    return {"loss": loss, "metric": metric}

  def on_validation_epoch_end(self):
    """
    LightningModule method called at the end of each validation epoch.
    """
    # Compute mean validation epoch loss and metric
    val_epoch_loss = torch.stack(self.val_step_loss).mean(0)
    val_epoch_metric = torch.stack(self.val_step_metric).mean(0) if len(self.val_step_metric) > 0 else None

    # Log and display the sum of validation epoch loss
    val_epoch_loss = torch.stack(self.val_step_loss).mean(0)

    output_names = self.trainer.datamodule.output_names

    # Log and display the sum of batch loss
    for i,loss_value in enumerate(val_epoch_loss):
      if isinstance(self.loss_fn, list):
        loss_name_i = self.loss_fn[i].name + '_' + output_names[i]
      else:
        loss_name_i = self.loss_fn.name + '_' + output_names[i]

      self.log(f"val_epoch_{loss_name_i}", val_epoch_loss[i], on_epoch=True, prog_bar=True)
    self.log(f"val_epoch_loss", sum(val_epoch_loss), on_epoch=True, prog_bar=False)
    #

    if self.track_performance:
      if self.val_history is None:
        self.current_val_epoch = 0
        self.val_history = {'epoch': torch.empty((0, 1)).to(device=val_epoch_loss.device, dtype=torch.long)}
        if self.track_performance:
            for i in range(self.model.num_outputs):
                metric_name_i = None
                if isinstance(self.loss_fn, list):
                  loss_name_i = self.loss_fn[i].name + '_' + output_names[i]
                  metric_name_i = self.metric_fn[i].name + '_' + output_names[i] if self.metric_fn[i] is not None else None
                else:
                  loss_name_i = self.loss_fn.name + '_' + output_names[i]
                  metric_name_i = self.metric_fn.name + '_' + output_names[i] if self.metric_fn is not None else None

                self.val_history[loss_name_i] = torch.empty((0, 1)).to(val_epoch_loss)

                if (val_epoch_metric is not None) & (metric_name_i is not None):
                    self.val_history[metric_name_i] = torch.empty((0, 1)).to(val_epoch_metric)

      else:
        self.val_history['epoch'] = torch.cat((self.val_history['epoch'],
                                                  torch.tensor(self.current_val_epoch).reshape(1, 1).to(val_epoch_loss)), 0)

        if self.track_performance:
            for i in range(self.trainer.datamodule.num_outputs):
              metric_name_i = None
              if isinstance(self.loss_fn, list):
                loss_name_i = self.loss_fn[i].name + '_' + output_names[i]
                metric_name_i = self.metric_fn[i].name + '_' + output_names[i] if self.metric_fn[i] is not None else None
              else:
                loss_name_i = self.loss_fn.name + '_' + output_names[i]
                metric_name_i = self.metric_fn.name + '_' + output_names[i] if self.metric_fn is not None else None

              self.val_history[loss_name_i] = torch.cat((self.val_history[loss_name_i],
                                                          val_epoch_loss[i].reshape(1, 1).to(val_epoch_loss)), 0)

              if val_epoch_metric is not None:
                self.val_history[metric_name_i] = torch.cat((self.val_history[metric_name_i],
                                                             val_epoch_metric[i].reshape(1, 1)))

        self.current_val_epoch += 1

    # Clear the validation epoch loss and metric lists
    self.val_step_loss.clear()
    self.val_step_metric.clear()
  ## End of validation

  ## Test Model
  def test_step(self, batch, batch_idx):
    """
    LightningModule method called during testing for each batch.
    """
    # Unpack batch
    input_batch, target_batch, steps_batch, batch_size, id = batch

    # Keep the first `batch_size` batches of hiddens
    if not self.stateful:
      self.hiddens = None
    elif self.hiddens is not None:
      for i in range(self.model.num_inputs):
        if (self.model.base_type[i] in ['gru', 'lstm', 'lru']) & (self.hiddens[i] is not None):
          if self.model.base_type[i] == 'lstm':
            if self.hiddens[i][0].shape[1] >= batch_size:
              self.hiddens[i] = tuple([s[:, :batch_size].contiguous() for s in self.hiddens[i]])
            else:
              self.hiddens[i] = tuple([torch.nn.functional.pad(s.contiguous(), pad=(0, 0, 0, batch_size-s.shape[1]), mode='constant', value=0) for s in self.hiddens[i]])
          else:
            if self.hiddens[i].shape[1] >= batch_size:
              self.hiddens[i] = self.hiddens[i][:, :batch_size].contiguous()
            else:
              self.hiddens[i] = torch.nn.functional.pad(self.hiddens[i].contiguous(), pad=(0, 0, 0, batch_size-self.hiddens[i].shape[1]), mode='constant', value=0)

    input_batch = input_batch[:batch_size]
    target_batch = target_batch[:batch_size]
    steps_batch = steps_batch[:batch_size]

    # Perform forward pass to compute gradients
    prediction_batch, self.hiddens = self.forward(input=input_batch,
                                                   steps=steps_batch,
                                                   hiddens=self.hiddens,
                                                   target=None,
                                                   input_window_idx=self.trainer.datamodule.test_input_window_idx,
                                                   output_window_idx=self.trainer.datamodule.test_output_window_idx,
                                                   output_input_idx=self.trainer.datamodule.output_input_idx,
                                                   input_output_idx=self.trainer.datamodule.input_output_idx,
                                                   output_mask=self.trainer.datamodule.test_output_mask)

    prediction_batch_masked = prediction_batch * self.trainer.datamodule.val_output_mask
    target_batch_masked = target_batch * self.trainer.datamodule.val_output_mask

    metric = None
    loss, metric = [], []
    j = 0
    for i in range(self.model.num_outputs):
      if isinstance(self.loss_fn, list):
        loss_fn_i = self.loss_fn[i]
        metric_fn_i = self.metric_fn[i]
      else:
        loss_fn_i = self.loss_fn
        metric_fn_i = self.metric_fn

      loss_i = sum(loss_fn_i(prediction_batch_masked[...,j:(j+self.model.output_size[i])],
                         target_batch_masked[...,j:(j+self.model.output_size[i])]))

      loss.append(loss_i.unsqueeze(0))

      if metric_fn_i is not None:
        metric_i = sum(metric_fn_i(prediction_batch_masked[...,j:(j+self.model.output_size[i])],
                               target_batch_masked[...,j:(j+self.model.output_size[i])]))
        metric.append(metric_i.unsqueeze(0))

      j += self.model.output_size[i]

    loss = torch.cat(loss)
    if metric is not None: metric = torch.cat(metric)

    # Store loss to be used later in `on_test_epoch_end`
    self.test_step_loss.append(loss)

    # Return loss and metric for the current batch
    return {"loss": loss, "metric": metric}

  def on_test_epoch_end(self):
    """
    LightningModule method called at the end of testing epoch.
    """
    # Calculate the mean of test epoch loss
    test_epoch_loss = torch.stack(self.test_epoch_loss).mean(0)

    # Calculate the mean of test epoch metric if available
    test_epoch_metric = torch.stack(self.test_epoch_metric).mean(0) if len(self.test_epoch_metric) > 0 else None

    # Clear the lists for the next epoch
    self.test_step_loss.clear()
    self.test_step_metric.clear()

    # Log the test epoch loss
    self.log('test_epoch_loss', test_epoch_loss.mean(), on_epoch=True, prog_bar=True)
  ## End of Testing

  ## plot history
  def plot_history(self,
                   history=None,
                   plot_train_history_by='epoch',
                   metric_digits = 4,
                   figsize=None):

    """
    Plot the training and validation history.

    Args:
        history (list, optional): List of parameter names to plot. Defaults to None.
        plot_train_history_by (str, optional): Choose whether to plot train history by 'epoch' or 'step'. Defaults to 'epoch'.
        figsize (tuple, optional): Figure size for the plot. Defaults to None.
    """

    output_names = self.trainer.datamodule.output_names

    # If history is not provided, use the loss function
    if history is None:
      if isinstance(self.loss_fn_name, list):
        history = [f"{fn.name}_{output_names[i]}" for i,fn in enumerate(self.loss_fn)]
      else:
        history = [f"{self.loss_fn.name}_{name}" for name in output_names]

    if plot_train_history_by == 'epoch':
      num_batches = len(self.trainer.datamodule.train_dl.dl)
      train_history_epoch = {'epoch': torch.arange(len(self.train_history['step']) // num_batches).to(dtype=torch.long)}
      num_epochs = len(train_history_epoch['epoch'])
      for key in self.train_history.keys():
        if key != 'step':
          batch_param = []
          for batch in self.train_history[key].split(num_batches, 0):
            batch_param.append(batch.mean(0, keepdim=True))
          batch_param = torch.cat(batch_param, 0)
          train_history_epoch[key] = batch_param[:num_epochs]

      train_history = train_history_epoch
      x_label = 'epoch'
    else:
      x_label = 'step'
      train_history = self.train_history

    num_params = len(history)
    fig = plt.figure(figsize=figsize if figsize is not None else (5, 5 * num_params))
    ax_i = 0
    for param in history:
      last_metric = None
      if any(name in param for name in output_names): # a loss or metric
        last_metric = np.round(train_history[param][-1].item(), metric_digits)

      ax_i += 1
      ax = fig.add_subplot(num_params, 1, ax_i)
      ax.plot(train_history[x_label].cpu(), train_history[param].cpu(), 'k',
              label = f"Train ({last_metric})" if last_metric is not None else 'Train')

      # Plot validation history if available and plotted by epochs
      if (self.val_history is not None) & (param in self.val_history) & (x_label == 'epoch'):
        N = np.min([self.val_history[x_label].shape[0], self.val_history[param].shape[0]])

        if any(name in param for name in output_names): # a loss or metric
          metric = self.val_history[param][:N]
          last_metric = np.round(metric[-1].item(), metric_digits)

        ax.plot(self.val_history[x_label][:N].cpu(), metric.cpu(), 'r',
                label = f"Val ({last_metric})" if last_metric is not None else 'Val')

      ax.set_title(param)
      ax.set_xlabel(x_label)
      ax.set_ylabel(param)
      ax.grid()
      ax.legend()
  ##

  ## Prediction
  def predict_step(self, batch, batch_idx):
    """
    Perform a prediction step for a batch of data.

    Args:
        batch (tuple): A tuple containing input_batch, target_batch, steps_batch, batch_size, and id.
        batch_idx (int): The index of the current batch.

    Returns:
        tuple: A tuple containing target_batch, prediction_batch, output_steps_batch, and id.
    """
    # Unpack batch
    input_batch, target_batch, steps_batch, batch_size, id = batch

    # Keep the first `batch_size` batches of hiddens
    if not self.stateful:
      self.hiddens = None
    elif self.hiddens is not None:
      for i in range(self.model.num_inputs):
        if (self.model.base_type[i] in ['gru', 'lstm', 'lru']) & (self.hiddens[i] is not None):
          if self.model.base_type[i] == 'lstm':
            if self.hiddens[i][0].shape[1] >= batch_size:
              self.hiddens[i] = [s[:, :batch_size].contiguous() for s in self.hiddens[i]]
            else:
              self.hiddens[i] = [torch.nn.functional.pad(s.contiguous(), pad=(0, 0, 0, batch_size-s.shape[1]), mode='constant', value=0) for s in self.hiddens[i]]
          else:
            if self.hiddens[i].shape[1] >= batch_size:
              self.hiddens[i] = self.hiddens[i][:, :batch_size].contiguous()
            else:
              self.hiddens[i] = torch.nn.functional.pad(self.hiddens[i].contiguous(), pad=(0, 0, 0, batch_size-self.hiddens[i].shape[1]), mode='constant', value=0)

    input_batch = input_batch[:batch_size]
    target_batch = target_batch[:batch_size]
    steps_batch = steps_batch[:batch_size]

    output_len = target_batch.shape[1]

    # Perform forward pass to compute predictions
    prediction_batch, self.hiddens = self.forward(input=input_batch,
                                                   steps=steps_batch,
                                                   hiddens=self.hiddens,
                                                   target=None,
                                                   input_window_idx=self.predict_input_window_idx,
                                                   output_window_idx=self.predict_output_window_idx,
                                                   output_input_idx=self.trainer.datamodule.output_input_idx,
                                                   input_output_idx=self.trainer.datamodule.input_output_idx,
                                                   output_mask=self.predict_output_mask)

    output_steps_batch = steps_batch[:, -output_len:]

    return target_batch, prediction_batch, output_steps_batch, id

  def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
    """
    Callback function called at the end of predicting a batch.

    Args:
        outputs (tuple): A tuple containing target_batch, prediction_batch, output_steps_batch, and id.
        batch: The batch data.
        batch_idx (int): The index of the current batch.
        dataloader_idx (int): The index of the dataloader used.

    Returns:
        None
    """
    self.step_target.append(outputs[0])
    self.prediction_batch.append(outputs[1])
    self.output_steps_batch.append(outputs[2])
    self.id.append(outputs[3])

  def on_predict_epoch_end(self):
    """
    Callback function called at the end of predicting an epoch.

    Returns:
        None
    """
    # Concatenate the lists to form tensors or arrays
    self.target = torch.cat(self.step_target, 0)  # Concatenate step targets
    self.prediction = torch.cat(self.prediction_batch, 0)  # Concatenate output predictions
    self.output_steps = torch.cat(self.output_steps_batch, 0)  # Concatenate output steps
    self.id = np.concatenate(self.id).tolist()  # Concatenate and convert to list

  def on_predict_epoch_start(self):
    """
    Callback function called at the start of predicting an epoch.

    Returns:
        None
    """
    # Initialize lists to store predictions, targets, output steps, and IDs for the epoch
    self.prediction_batch, self.step_target = [], []  # Initialize lists for predictions and targets
    self.output_steps_batch = []  # Initialize list for output steps
    self.id = []  # Initialize list for IDs
    # self.hiddens = None

  def predict(self, reduction='mean'):
    """
    Perform predictions on training, validation, and test datasets.

    Args:
      reduction (str): Reduction method to apply to predictions ('mean', 'sum', 'none').

    Returns:
      None
    """

    # Move the model to GPU if applicable
    if self.accelerator == 'gpu':
      self.model.to('cuda')

    # Set predicting mode for datamodule
    self.trainer.datamodule.predicting = True

    # Disable progress bar during prediction
    self.trainer.enable_progress_bar = False

    # Determine starting step based on whether data is padded
    start_step = self.trainer.datamodule.start_step if self.trainer.datamodule.pad_data else 0

    # Retrieve data and transforms from datamodule
    data = self.trainer.datamodule.data
    transforms = self.trainer.datamodule.transforms

    # Split data into train, validation, and test datasets
    train_data, val_data, test_data = self.trainer.datamodule.train_data, self.trainer.datamodule.val_data, self.trainer.datamodule.test_data

    # If there's only one dataset, convert to a list for consistency
    if self.trainer.datamodule.num_datasets == 1:
      data = [data]
      transforms = [transforms]

    # Retrieve time name and output size from datamodule
    time_name = self.trainer.datamodule.time_name
    output_size = self.trainer.datamodule.output_size

    # Retrieve IDs for all datasets
    ids = [data_['id'] for data_ in data]

    with torch.no_grad():
      self.hiddens = None  # Reset hidden states

      # Predict on training data
      if not isinstance(train_data, list):
        train_data = [train_data]
      train_ids = [data_['id'] for data_ in train_data]

      self.predict_output_mask = self.trainer.datamodule.train_output_mask
      self.predict_input_window_idx = self.trainer.datamodule.train_input_window_idx
      self.predict_output_window_idx = self.trainer.datamodule.train_output_window_idx
      
      shuffle_train_original = self.trainer.datamodule.shuffle_train
      if shuffle_train_original == True:
        print("Unshuffling training data")
        self.trainer.datamodule.shuffle_train = False
        self.trainer.datamodule.predicting = False
        self.trainer.datamodule.train_dataloader()
        print("Unshuffling complete")

      self.trainer.predict(self, self.trainer.datamodule.train_dl.dl)
      self.trainer.datamodule.shuffle_train = shuffle_train_original
      self.trainer.datamodule.predicting = True

      self.train_prediction_data = [[] for _ in range(len(train_data))]

      # Process predictions for each unique ID
      for id in list(np.unique(self.id)):

        data_idx = ids.index(id)
        train_idx = train_ids.index(id)

        self.train_prediction_data[train_idx] = {'id': id}

        sample_idx = [idx for idx, value in enumerate(self.id) if value == id]
        transform_idx = transforms[data_idx]

        prediction, target, output_steps = self.prediction[sample_idx], self.target[sample_idx], self.output_steps[sample_idx]

        if prediction.ndim < 3:
          prediction, target, output_steps = prediction.unsqueeze(0), target.unsqueeze(0), output_steps.unsqueeze(0)

        train_prediction, train_output_steps = self.generate_reduced_output(prediction, output_steps,
                                                                            reduction=reduction,
                                                                            transforms=transform_idx)

        train_target, _ = self.generate_reduced_output(target, output_steps,
                                                       reduction=reduction,
                                                       transforms=transform_idx)

        # Retrieve train time and output steps
        if self.trainer.datamodule.num_datasets > 1:
          train_time = train_data[train_idx][time_name]
          train_output_steps = train_output_steps.cpu().numpy()
        else:
          train_time = data[data_idx][time_name]
          train_output_steps = train_output_steps.cpu().numpy() - start_step

        if hasattr(train_time, 'tz'):
          train_time = train_time.dt.tz_localize(None).values

        self.train_prediction_data[train_idx][time_name] = train_time[train_output_steps][start_step:]

        j = 0
        for i, output_name in enumerate(self.trainer.datamodule.output_names):
          train_target_i = train_target[:, j:(j + output_size[i])]
          train_prediction_i = train_prediction[:, j:(j + output_size[i])]

          self.train_prediction_data[train_idx][f"{output_name}_target"] = train_target_i[start_step:]
          self.train_prediction_data[train_idx][f"{output_name}_prediction"] = train_prediction_i[start_step:]

          # Compute global loss for each output
          metric_name_i = None
          if isinstance(self.loss_fn, list):
            loss_name_i = self.loss_fn[i].name
            metric_name_i = self.metric_fn[i].name if self.metric_fn[i] is not None else None
          else:
            loss_name_i = self.loss_fn.name
            metric_name_i = self.metric_fn.name if self.metric_fn is not None else None

          train_loss_i = Criterion(loss_name_i, dims=(0, 1))(train_prediction_i.unsqueeze(0), train_target_i.unsqueeze(0))
          self.train_prediction_data[train_idx][f"{output_name}_global_{loss_name_i}"] = train_loss_i

          # Compute step-wise loss for each output
          train_loss_i = Criterion(loss_name_i, dims=0)(train_prediction_i.unsqueeze(0), train_target_i.unsqueeze(0))
          self.train_prediction_data[train_idx][f"{output_name}_step_{loss_name_i}"] = train_loss_i

          if metric_name_i is not None:
            # Compute global metric for each output
            train_metric_i = Criterion(metric_name_i, dims=(0, 1))(train_prediction_i.unsqueeze(0), train_target_i.unsqueeze(0))
            self.train_prediction_data[train_idx][f"{output_name}_global_{metric_name_i}"] = train_metric_i

            # Compute step-wise metric for each output
            train_metric_i = Criterion(metric_name_i, dims=0)(train_prediction_i.unsqueeze(0), train_target_i.unsqueeze(0))
            self.train_prediction_data[train_idx][f"{output_name}_step_{metric_name_i}"] = train_metric_i

          j += output_size[i]

      # Predict on validation data
      self.val_prediction_data = None
      if len(val_data) > 0:

        if not isinstance(val_data, list):
          val_data = [val_data]
        val_ids = [data_['id'] for data_ in val_data]

        self.predict_output_mask = self.trainer.datamodule.val_output_mask
        self.predict_input_window_idx = self.trainer.datamodule.val_input_window_idx
        self.predict_output_window_idx = self.trainer.datamodule.val_output_window_idx

        self.trainer.predict(self, self.trainer.datamodule.val_dl.dl)

        self.val_prediction_data = [[] for _ in range(len(val_data))]

        # Process predictions for each unique ID
        for id in list(np.unique(self.id)):

            data_idx = ids.index(id)
            val_idx = val_ids.index(id)

            self.val_prediction_data[val_idx] = {'id': id}

            sample_idx = [idx for idx, value in enumerate(self.id) if value == id]
            transform_idx = transforms[data_idx]

            prediction, target, output_steps = self.prediction[sample_idx], self.target[sample_idx], self.output_steps[sample_idx]

            if prediction.ndim < 3:
                prediction, target, output_steps = prediction.unsqueeze(0), target.unsqueeze(0), output_steps.unsqueeze(0)

            val_prediction, val_output_steps = self.generate_reduced_output(prediction, output_steps,
                                                                            reduction = reduction,
                                                                            transforms = transform_idx)

            val_target, _ = self.generate_reduced_output(target, output_steps,
                                                         reduction=reduction,
                                                         transforms=transform_idx)

            # Retrieve validation time and output steps
            if self.trainer.datamodule.num_datasets > 1:
              val_time = val_data[val_idx][time_name]
              val_output_steps = val_output_steps.cpu().numpy()
            else:
              val_time = data[data_idx][time_name]
              val_output_steps = val_output_steps.cpu().numpy() - start_step

            if hasattr(val_time, 'tz'):
              val_time = val_time.dt.tz_localize(None).values

            self.val_prediction_data[val_idx][time_name] = val_time[val_output_steps]

            j = 0
            for i, output_name in enumerate(self.trainer.datamodule.output_names):
              val_target_i = val_target[:, j:(j + output_size[i])]
              val_prediction_i = val_prediction[:, j:(j + output_size[i])]

              self.val_prediction_data[val_idx][f"{output_name}_target"] = val_target_i
              self.val_prediction_data[val_idx][f"{output_name}_prediction"] = val_prediction_i

              # Compute global loss for each output
              metric_name_i = None
              if isinstance(self.loss_fn, list):
                loss_name_i = self.loss_fn[i].name
                metric_name_i = self.metric_fn[i].name if self.metric_fn[i] is not None else None
              else:
                loss_name_i = self.loss_fn.name
                metric_name_i = self.metric_fn.name if self.metric_fn is not None else None

              val_loss_i = Criterion(loss_name_i, dims=(0, 1))(val_prediction_i.unsqueeze(0), val_target_i.unsqueeze(0))
              self.val_prediction_data[val_idx][f"{output_name}_global_{loss_name_i}"] = val_loss_i

              # Compute step-wise loss for each output
              val_loss_i = Criterion(loss_name_i, dims=0)(val_prediction_i.unsqueeze(0), val_target_i.unsqueeze(0))
              self.val_prediction_data[val_idx][f"{output_name}_step_{loss_name_i}"] = val_loss_i

              if metric_name_i is not None:
                # Compute global metric for each output
                val_metric_i = Criterion(metric_name_i, dims=(0, 1))(val_prediction_i.unsqueeze(0), val_target_i.unsqueeze(0))
                self.val_prediction_data[val_idx][f"{output_name}_global_{metric_name_i}"] = val_metric_i

                # Compute step-wise metric for each output
                val_metric_i = Criterion(metric_name_i, dims=0)(val_prediction_i.unsqueeze(0), val_target_i.unsqueeze(0))
                self.val_prediction_data[val_idx][f"{output_name}_step_{metric_name_i}"] = val_metric_i

              j += output_size[i]

      # Predict on test data
      self.test_prediction_data = None
      if len(test_data) > 0:
        # If test dataloader is not already initialized, initialize it
        if not hasattr(self.trainer.datamodule, 'test_dl'):
          self.trainer.datamodule.test_dataloader()

        if not isinstance(test_data, list):
          test_data = [test_data]
        test_ids = [data_['id'] for data_ in test_data]

        self.predict_output_mask = self.trainer.datamodule.test_output_mask
        self.predict_input_window_idx = self.trainer.datamodule.test_input_window_idx
        self.predict_output_window_idx = self.trainer.datamodule.test_output_window_idx

        self.trainer.predict(self, self.trainer.datamodule.test_dl.dl)

        self.test_prediction_data = [[] for _ in range(len(test_data))]

        # Process predictions for each unique ID
        for id in list(np.unique(self.id)):

          data_idx = ids.index(id)
          test_idx = test_ids.index(id)

          self.test_prediction_data[test_idx] = {'id': id}

          sample_idx = [idx for idx, value in enumerate(self.id) if value == id]
          transform_idx = transforms[data_idx]

          prediction, target, output_steps = self.prediction[sample_idx], self.target[sample_idx], self.output_steps[sample_idx]

          if prediction.ndim < 3:
            prediction, target, output_steps = prediction.unsqueeze(0), target.unsqueeze(0), output_steps.unsqueeze(0)

          test_prediction, test_output_steps = self.generate_reduced_output(prediction, output_steps,
                                                                            reduction=reduction,
                                                                            transforms=transform_idx)

          test_target, _ = self.generate_reduced_output(target, output_steps,
                                                        reduction = reduction,
                                                        transforms = transform_idx)

          # Retrieve test time and output steps
          if self.trainer.datamodule.num_datasets > 1:
            test_time = test_data[test_idx][time_name]
            test_output_steps = test_output_steps.cpu().numpy()
          else:
            test_time = data[data_idx][time_name]
            test_output_steps = test_output_steps.cpu().numpy() - start_step

          if hasattr(test_time, 'tz'):
            test_time = test_time.dt.tz_localize(None).values

          self.test_prediction_data[test_idx][time_name] = test_time[test_output_steps]

          j = 0
          for i, output_name in enumerate(self.trainer.datamodule.output_names):
            test_target_i = test_target[:, j:(j + output_size[i])]
            test_prediction_i = test_prediction[:, j:(j + output_size[i])]

            self.test_prediction_data[test_idx][f"{output_name}_target"] = test_target_i
            self.test_prediction_data[test_idx][f"{output_name}_prediction"] = test_prediction_i

            # Compute global loss for each output
            metric_name_i = None
            if isinstance(self.loss_fn, list):
              loss_name_i = self.loss_fn[i].name
              metric_name_i = self.metric_fn[i].name if self.metric_fn[i] is not None else None
            else:
              loss_name_i = self.loss_fn.name
              metric_name_i = self.metric_fn.name if self.metric_fn is not None else None

            test_loss_i = Criterion(loss_name_i, dims=(0, 1))(test_prediction_i.unsqueeze(0), test_target_i.unsqueeze(0))
            self.test_prediction_data[test_idx][f"{output_name}_global_{loss_name_i}"] = test_loss_i

            # Compute step-wise loss for each output
            test_loss_i = Criterion(loss_name_i, dims=0)(test_prediction_i.unsqueeze(0), test_target_i.unsqueeze(0))
            self.test_prediction_data[test_idx][f"{output_name}_step_{loss_name_i}"] = test_loss_i

            if metric_name_i is not None:
              # Compute global metric for each output
              test_metric_i = Criterion(metric_name_i, dims=(0, 1))(test_prediction_i.unsqueeze(0), test_target_i.unsqueeze(0))
              self.test_prediction_data[test_idx][f"{output_name}_global_{metric_name_i}"] = test_metric_i

              # Compute step-wise metric for each output
              test_metric_i = Criterion(metric_name_i, dims=0)(test_prediction_i.unsqueeze(0), test_target_i.unsqueeze(0))
              self.test_prediction_data[test_idx][f"{output_name}_step_{metric_name_i}"] = test_metric_i

            j += output_size[i]

    # If there's only one dataset, convert to a single dictionary
    if len(data) == 1:
      self.train_prediction_data = self.train_prediction_data[0]
      if self.val_prediction_data is not None:
        self.val_prediction_data = self.val_prediction_data[0]
      if self.test_prediction_data is not None:
        self.test_prediction_data = self.test_prediction_data[0]

    # Reset progress bar and predicting mode
    self.trainer.enable_progress_bar = True
    self.trainer.datamodule.predicting = False

  ##
  def plot_predictions(self,
                       id = None,
                       output_feature_units = None,
                       include_baseline = False,
                       figsize = None):

    time_name = self.trainer.datamodule.time_name
    output_names = self.trainer.datamodule.output_names
    output_feature_names = self.trainer.datamodule.output_feature_names
    num_outputs = len(output_names)
    output_size = self.trainer.datamodule.output_size
    max_output_size = np.max(output_size)

    start_step = self.trainer.datamodule.start_step

    if self.trainer.datamodule.num_datasets == 1:

      rows, cols = num_outputs, max_output_size
      fig, ax = plt.subplots(rows, cols, figsize = figsize if figsize is not None else (20*max_output_size, 5*num_outputs))

      train_time = self.train_prediction_data[time_name]
      if hasattr(train_time, 'tz'): train_time = train_time.dt.tz_localize(None)

      if self.val_prediction_data is not None:
        val_time = self.val_prediction_data[time_name]
        if hasattr(val_time, 'tz'): val_time = val_time.dt.tz_localize(None)

      else:
        val_time = None

      if self.test_prediction_data is not None:
        test_time = self.test_prediction_data[time_name]
        if hasattr(test_time, 'tz'): test_time = test_time.dt.tz_localize(None)

      else:
        test_time = None

      for i,output_name in enumerate(output_names):

        try:
          ax_i = ax[i, :]
          [ax_j.axis("off") for ax_j in ax_i]
        except:
          pass

        for f in range(output_size[i]):

          if (output_feature_names is not None) & (output_size[i] > 1):
            output_feature_name_if = output_feature_names[f]
          else:
            output_feature_name_if = None

          if output_feature_units is not None:
            if output_name in output_feature_units:
              output_feature_units_if = output_feature_units[output_name][f]
            else:
              output_feature_units_if = None
          else:
            output_feature_units_if = None

          try:
            ax_if = ax[i,f]
          except:
            try:
              j = i if (rows>1) & (cols == 1) else f
              ax_if = ax[j]
            except:
              ax_if = ax

          train_target_if = self.train_prediction_data[f"{output_name}_target"][:, f].cpu()
          train_prediction_if = self.train_prediction_data[f"{output_name}_prediction"][:, f].cpu()
          train_loss_if = np.round(self.train_prediction_data[f"{output_name}_global_{self.loss_fn.name}"][f].item(), 2)
          train_metric_if = np.round(self.train_prediction_data[f"{output_name}_global_{self.metric_fn.name}"][f].item(), 2) if self.metric_fn is not None else None
          if include_baseline:
            train_baseline_prediction_if = self.train_prediction_data[f"{output_name}_baseline_prediction"][:, f].cpu()
            train_baseline_loss_if = np.round(self.train_prediction_data[f"{output_name}_baseline_{self.loss_fn.name}"][f].item(),2)
            train_baseline_metric_if = np.round(self.train_prediction_data[f"{output_name}_baseline_{self.metric_fn.name}"][f].item(),2) if self.metric_fn is not None else None

          ax_if.plot(train_time, train_target_if, '-k', label = 'Actual')
          ax_if.plot(train_time, train_prediction_if, '-r', label = 'Prediction')
          train_label = f"Train ({self.loss_fn.name} = {train_loss_if}, {self.metric_fn.name} = {train_metric_if})" \
                        if train_metric_if is not None \
                        else f"Train ({self.loss_fn.name} = {train_loss_if})"
          if include_baseline:
            ax_if.plot(train_time, train_baseline_prediction_if, '--g', linewidth = 1.0, label = 'Baseline')
            train_label = train_label + f", Baseline ({self.loss_fn.name} = {train_baseline_loss_if}, {self.metric_fn.name} = {train_baseline_metric_if})"

          ax_if.axvspan(train_time.values.min(), train_time.values.max(), facecolor='gray', alpha=0.2, label = train_label)

          if val_time is not None:
            val_target_if = self.val_prediction_data[f"{output_name}_target"][:, f].cpu()
            val_prediction_if = self.val_prediction_data[f"{output_name}_prediction"][:, f].cpu()
            val_loss_if = np.round(self.val_prediction_data[f"{output_name}_global_{self.loss_fn.name}"][f].item(),2)
            val_metric_if = np.round(self.val_prediction_data[f"{output_name}_global_{self.metric_fn.name}"][f].item(),2) if self.metric_fn is not None else None
            if include_baseline:
              val_baseline_prediction_if = self.val_prediction_data[f"{output_name}_baseline_prediction"][:, f].cpu()
              val_baseline_loss_if = np.round(self.val_prediction_data[f"{output_name}_baseline_{self.loss_fn.name}"][f].item(),2)
              val_baseline_metric_if = np.round(self.val_prediction_data[f"{output_name}_baseline_{self.metric_fn.name}"][f].item(),2) if self.metric_fn is not None else None

            ax_if.plot(val_time, val_target_if, '-k')
            ax_if.plot(val_time, val_prediction_if, '-r')
            val_label = f"Val ({self.loss_fn.name} = {val_loss_if}, {self.metric_fn.name} = {val_metric_if})" \
                          if val_metric_if is not None \
                          else f"Val ({self.loss_fn.name} = {val_loss_if})"
            if include_baseline:
              ax_if.plot(val_time, val_baseline_prediction_if, '--g', linewidth = 1.0)
              val_label = val_label + f", Baseline ({self.loss_fn.name} = {val_baseline_loss_if}, {self.metric_fn.name} = {val_baseline_metric_if})"

            ax_if.axvspan(val_time.values.min(), val_time.values.max(), facecolor='blue', alpha=0.2, label = val_label)

          if test_time is not None:
            test_target_if = self.test_prediction_data[f"{output_name}_target"][:, f].cpu()
            test_prediction_if = self.test_prediction_data[f"{output_name}_prediction"][:, f].cpu()
            test_loss_if = np.round(self.test_prediction_data[f"{output_name}_global_{self.loss_fn.name}"][f].item(),2)
            test_metric_if = np.round(self.test_prediction_data[f"{output_name}_global_{self.metric_fn.name}"][f].item(),2) if self.metric_fn is not None else None
            if include_baseline:
              test_baseline_prediction_if = self.test_prediction_data[f"{output_name}_global_baseline_prediction"][:, f].cpu()
              test_baseline_loss_if = np.round(self.test_prediction_data[f"{output_name}_global_baseline_{self.loss_fn.name}"][f].item(),2)
              test_baseline_metric_if = np.round(self.test_prediction_data[f"{output_name}_global_baseline_{self.metric_fn.name}"][f].item(),2) if self.metric_fn is not None else None

            ax_if.plot(test_time, test_target_if, '-k')
            ax_if.plot(test_time, test_prediction_if, '-r')
            test_label = f"Test ({self.loss_fn.name} = {test_loss_if}, {self.metric_fn.name} = {test_metric_if})" \
                          if test_metric_if is not None \
                          else f"Test ({self.loss_fn.name} = {test_loss_if})"
            if include_baseline:
              ax_if.plot(test_time, test_baseline_prediction_if, '--g', linewidth = 1.0)
              test_label = test_label + f", Baseline ({self.loss_fn.name} = {test_baseline_loss_if}, {self.metric_fn.name} = {test_baseline_metric_if})"

            ax_if.axvspan(test_time.values.min(), test_time.values.max(), facecolor='red', alpha=0.2, label = test_label)

          if (f == 0) & (output_feature_names is not None):
            ax_if.set_title(output_name)
          if f == output_size[i] - 1:
            ax_if.set_xlabel(f"Time [{self.trainer.datamodule.time_unit}]")

          if output_feature_names is None:
            ylabel = f"{output_name} [{output_feature_units_if}]" if output_feature_units_if is not None else f"{output_name}"
          elif output_feature_name_if is not None:
            ylabel = f"{output_feature_name_if} [{output_feature_units_if}]" if output_feature_units_if is not None else f"{output_feature_name_if}"
          else:
            ylabel = f"[{output_feature_units_if}]" if output_feature_units_if is not None else None

          ax_if.set_ylabel(ylabel)

          ax_if.legend(loc = 'upper left', bbox_to_anchor = (1.02, 1), ncol = 1) # loc = 'upper center', bbox_to_anchor = (0.5, 1.15), ncol = 5))
          ax_if.grid()

      if num_outputs > 1:
        for i in range(num_outputs, rows):
            ax[i].axis("off")

    else:

      id = id or list(self.trainer.datamodule.data)[0]['id']

      data_idx = [idx for idx,data in enumerate(list(self.train_prediction_data)) if list(self.train_prediction_data)[idx]['id'] == id]
      if len(data_idx) > 0:
        prediction_data = list(self.train_prediction_data)[data_idx[0]]
        group = 'Train'
      if (self.val_prediction_data is not None) & (len(data_idx) == 0):
        data_idx = [idx for idx,data in enumerate(list(self.val_prediction_data)) if list(self.val_prediction_data)[idx]['id'] == id]
        if len(data_idx) > 0:
          prediction_data = list(self.val_prediction_data)[data_idx[0]]
        group = 'Val'
      if (list(self.test_prediction_data) is not None) & (len(data_idx) == 0):
        data_idx = [idx for idx,data in enumerate(list(self.test_prediction_data)) if list(self.test_prediction_data)[idx]['id'] == id]
        if len(data_idx) > 0:
          prediction_data = list(self.test_prediction_data)[data_idx[0]]
        group = 'Test'

      rows, cols = num_outputs, max_output_size
      fig, ax = plt.subplots(rows, cols, figsize = figsize if figsize is not None else (20*max_output_size, 5*num_outputs))

      time = prediction_data[time_name]
      if hasattr(time, 'tz'): time = time.dt.tz_localize(None)

      for i,output_name in enumerate(output_names):

        if isinstance(self.loss_fn, list):  
          loss_fn_i, metric_fn_i = self.loss_fn[i], self.metric_fn[i]
        else:
          loss_fn_i, metric_fn_i = self.loss_fn, self.metric_fn

        try:
          ax_i = ax[i, :]
          [ax_j.axis("off") for ax_j in ax_i]
        except:
          pass

        for f in range(output_size[i]):

          if (output_feature_names is not None) & (output_size[i] > 1):
            output_feature_name_if = output_feature_names[f]
          else:
            output_feature_name_if = None

          if output_feature_units is not None:
            if output_name in output_feature_units:
              output_feature_units_if = output_feature_units[output_name][f]
            else:
              output_feature_units_if = None
          else:
            output_feature_units_if = None

          try:
            ax_if = ax[i,f]
          except:
            try:
              j = i if (rows>1) & (cols == 1) else f
              ax_if = ax[j]
            except:
              ax_if = ax

          target_if = prediction_data[f"{output_name}_target"][:, f].cpu()
          prediction_if =prediction_data[f"{output_name}_prediction"][:, f].cpu()
          loss_if = np.round(prediction_data[f"{output_name}_global_{loss_fn_i.name}"][f].item(), 2)
          metric_if = np.round(prediction_data[f"{output_name}_global_{metric_fn_i.name}"][f].item(), 2) if metric_fn_i is not None else None
          if include_baseline:
            baseline_prediction_if = prediction_data[f"{output_name}_baseline_prediction"][:, f].cpu()
            baseline_loss_if = np.round(prediction_data[f"{output_name}_baseline_{loss_fn_i.name}"][f].item(),2)
            baseline_metric_if = np.round(prediction_data[f"{output_name}_baseline_{metric_fn_i.name}"][f].item(),2) if metric_fn_i is not None else None

          ax_if.plot(time, target_if, '-k', label = 'Actual')
          ax_if.plot(time, prediction_if, '-r', label = 'Prediction')

          output_name_ = output_name.upper() if output_size[i] == 1 else f"{output_name.upper()}:{output_feature_names[f]}"

          title = f"ID {id} ({group}): {output_name_} | {loss_fn_i.name} = {loss_if}, {metric_fn_i.name} = {metric_if}" \
                    if metric_if is not None \
                    else f"{output_name_} ({loss_fn_i.name} = {loss_if})"
          if include_baseline:
            ax_if.plot(time, baseline_prediction_if, '--g', linewidth = 1.0, label = 'Baseline')
            title = title + f"| Baseline: {loss_fn_i.name} = {baseline_loss_if}, {metric_fn_i.name} = {baseline_metric_if}"

          ax_if.set_title(title)

          if f == output_size[i] - 1:
            ax_if.set_xlabel(f"Time [{self.trainer.datamodule.time_unit}]")

          if output_feature_names is None:
            ylabel = f"{output_name} [{output_feature_units_if}]" if output_feature_units_if is not None else f"{output_name}"
          elif output_feature_name_if is not None:
            ylabel = f"{output_feature_name_if} [{output_feature_units_if}]" if output_feature_units_if is not None else f"{output_feature_name_if}"
          else:
            ylabel = f"[{output_feature_units_if}]" if output_feature_units_if is not None else None

          ax_if.set_ylabel(ylabel)

          ax_if.legend(loc = 'upper left', bbox_to_anchor = (1.02, 1), ncol = 1) # loc = 'upper center', bbox_to_anchor = (0.5, 1.15), ncol = 5))
          ax_if.grid()

      if num_outputs > 1:
        for i in range(num_outputs, rows):
            ax[i].axis("off")

    fig.tight_layout()

    self.actual_prediction_ax = ax
    self.actual_prediction_plot = plt.gcf()

    # return fig
  ##

  ##
  def evaluate_model(self, metric):
    """
    Evaluate the model's performance using specified loss and metric on test, validation, or training data.

    Args:
        metric (str): function to evaluate ('mse', 'mae', etc.).

    Returns:
        None
    """

    self.eval_metric = metric

    # Move the model to GPU if applicable
    if self.accelerator == 'gpu':
      self.model.to('cuda')

    metric_name = metric

    time_name = self.trainer.datamodule.time_name

    stride = self.trainer.datamodule.stride

    # Initialize metric function and metric function
    metric_fn = Criterion(metric_name)

    # Select prediction data based on availability
    if self.test_prediction_data is not None:
      prediction_data = self.test_prediction_data
    elif self.val_prediction_data is not None:
      prediction_data = self.val_prediction_data
    else:
      prediction_data = self.train_prediction_data

    if not isinstance(prediction_data, list):
      prediction_data = [prediction_data]

    self.evaluation_data = []

    # Iterate through prediction data
    for data_idx in range(len(prediction_data)):

      time = prediction_data[data_idx][time_name]

      self.evaluation_data.append({})

      self.evaluation_data[data_idx][time_name] = time

      for name in self.trainer.datamodule.output_names:

        target = prediction_data[data_idx][f"{name}_target"]
        prediction = prediction_data[data_idx][f"{name}_prediction"]

        # Calculate metric
        step_metric = metric_fn(target, prediction)
        global_metric = step_metric.mean(0)
        stride_metric, stride_time = [], []

        self.evaluation_data[data_idx][f"{name}_step_{metric_name}"] = step_metric
        self.evaluation_data[data_idx][f"{name}_global_{metric_name}"] = global_metric

        # Calculate metric over strides
        for i in range(stride, step_metric.shape[0] + 1, stride):
            stride_time.append(time[(i - stride):i])
            stride_metric.append(step_metric[(i - stride):i].mean(0))

        self.evaluation_data[data_idx][f"{name}_stride_{metric_name}"] = torch.cat(stride_metric, 0)

      self.evaluation_data[data_idx][f"stride_{self.trainer.datamodule.time_name}"] = stride_time

    # If there's only one dataset, convert to a single dictionary
    if len(self.evaluation_data) == 1:
      self.evaluation_data = self.evaluation_data[0]
  ##

  ##
  def plot_evaluation(self,
                    index = 'step',
                    density = True,
                    num_bins = None,
                    figsize = None):

    evaluation_data = self.evaluation_data
    output_feature_names = self.trainer.datamodule.output_feature_names
    output_feature_size = self.trainer.datamodule.output_feature_size
    num_output_features = len(output_feature_names)
    time_name = self.trainer.datamodule.time_name

    figsize = figsize or (20, 5*num_output_features)
    num_bins = num_bins or 20

    step_time = evaluation_data[time_name]

    fig, ax = plt.subplots(num_output_features, 2, figsize = figsize)

    if index == 'stride':

      stride_time = evaluation_data[f"stride_{time_name}"]

      num_bins = num_bins or 20

      for i in range(num_output_features):

        stride_metric_i = evaluation_data[f"{output_feature_names[i]}_stride_{self.eval_metric}"].cpu()

        if num_output_features > 1:
          ax0_i, ax1_i = ax[i,0], ax[i,1]
        else:
          ax0_i, ax1_i = ax[0], ax[1]

        for j,t in enumerate(stride_time):
          ax0_i.plot(t, stride_metric_i[j].repeat(len(t)), '-b')

        ax0_i.set_title(f"Stride-wise {self.eval_metric.upper()}")
        ax0_i.set_xlabel(time_name)
        ax0_i.grid()

        ax1_i.hist(stride_metric_i, bins = num_bins, edgecolor = 'k', alpha = 0.7, density = density)
        ax1_i.set_xlabel(self.eval_metric.upper())
        ax1_i.set_ylabel('Frequency' if density else 'Count')
        ax1_i.grid()

    else:

      for i in range(num_output_features):

        step_metric_i = evaluation_data[f"{output_feature_names[i]}_step_{self.eval_metric}"].cpu().squeeze()

        if num_output_features > 1:
          ax0_i, ax1_i = ax[i,0], ax[i,1]
        else:
          ax0_i, ax1_i = ax[0], ax[1]

        ax0_i.plot(step_time, step_metric_i, '-.b')
        ax0_i.set_title(f"Step-wise {self.eval_metric.upper()}")
        ax0_i.set_xlabel(time_name)

        ax0_i.grid()

        ax1_i.hist(step_metric_i, bins = num_bins, edgecolor = 'k', alpha = 0.7, density = density)
        ax1_i.set_xlabel(self.eval_metric.upper())
        ax1_i.set_ylabel('Frequency' if density else 'Count')
        ax1_i.grid()

    fig.tight_layout()
  ##

  ##
  def plot_residuals(self,
                     id = None,
                     figsize = None,
                     num_bins = None,
                     density = True):

    data = self.trainer.datamodule.data
    if not isinstance(data, list): data = [data]

    id = id or data[0]['id']

    train_data = self.train_prediction_data
    if not isinstance(train_data, list): train_data = [train_data]

    data_idx = [idx for idx,data in enumerate(train_data) if train_data[idx]['id'] == id]
    if len(data_idx) > 0:
      prediction_data = train_data[data_idx[0]]
      group = 'Train'
    if (self.val_prediction_data is not None) & (len(data_idx) == 0):
      val_data = self.val_prediction_data
      if not isinstance(val_data, list): val_data = [val_data]
      data_idx = [idx for idx,data in enumerate(val_data) if val_data[idx]['id'] == id]
      if len(data_idx) > 0:
        prediction_data = val_data[data_idx[0]]
      group = 'Val'
    if (self.test_prediction_data is not None) & (len(data_idx) == 0):
      test_data = self.test_prediction_data
      if not isinstance(test_data, list): test_data = [test_data]
      data_idx = [idx for idx,data in enumerate(test_data) if test_data[idx]['id'] == id]
      if len(data_idx) > 0:
        prediction_data = test_data[data_idx[0]]
      group = 'Test'

    output_names = self.trainer.datamodule.output_names
    num_outputs = self.trainer.datamodule.num_outputs
    time_unit = self.trainer.datamodule.time_unit

    figsize = figsize or (20, 5*num_outputs)

    time = prediction_data[self.trainer.datamodule.time_name]

    num_bins = num_bins or 20

    fig, ax = plt.subplots(num_outputs, 2, figsize = figsize)

    for i, output_name in enumerate(output_names):
      residual_i = (prediction_data[f"{output_name}_target"] - prediction_data[f"{output_name}_prediction"]).cpu()

      if num_outputs > 1:
        ax0_i, ax1_i = ax[i,0], ax[i,1]
      else:
        ax0_i, ax1_i = ax[0], ax[1]

      ax0_i.plot(time, residual_i, 'k')
      ax0_i.grid()
      ax0_i.set_xlabel(f"Time [{time_unit}]")
      ax0_i.set_ylabel(output_name)
      ax0_i.set_title(f"ID {id} ({group}): {output_name.upper()}")

      ax1_i.hist(residual_i.t(), bins = num_bins, edgecolor = 'k', alpha = 0.7, density = density)
      ax1_i.set_xlabel('Deviation')
      ax1_i.set_ylabel('Frequency' if density else 'Count')
      ax1_i.grid()

    fig.tight_layout()
  ##

  ## forecast
  def forecast(self,
               num_forecast_steps = None,
               id = None,
               hiddens = None,
               invert = True,
               eval = False):

    data, transforms = self.trainer.datamodule.data, self.trainer.datamodule.transforms
    if not isinstance(data, list):
      data = [data]
      transforms = [transforms]

    id = id or data[0]['id']

    idx = [idx for idx,data_ in enumerate(data) if data_['id'] == id][0]

    data, transforms = data[idx], transforms[idx]

    # Fetch relevant names and sizes from the datamodule
    time_name = self.trainer.datamodule.time_name
    input_names = self.trainer.datamodule.input_names
    output_names = self.trainer.datamodule.output_names
    output_feature_names = self.trainer.datamodule.output_feature_names
    output_feature_size = self.trainer.datamodule.output_feature_size

    # Move the model to the appropriate device and data type
    self.model.to(device = self.trainer.datamodule.device,
                  dtype = self.trainer.datamodule.dtype)

    # Extract various indices and values
    input_window_idx = self.trainer.datamodule.train_input_window_idx
    output_window_idx = self.trainer.datamodule.train_output_window_idx
    total_window_size = self.trainer.datamodule.train_dl.total_window_size
    # max_input_len = self.trainer.datamodule.train_max_input_len
    # max_output_len = self.trainer.datamodule.train_max_output_len
    total_input_len = len(torch.cat(input_window_idx).unique())
    total_output_len = len(torch.cat(output_window_idx).unique())
    total_input_size = sum(self.trainer.datamodule.input_size)
    output_size = self.trainer.datamodule.output_size
    total_output_size = sum(output_size)
    output_mask = self.trainer.datamodule.train_output_mask
    output_input_idx, input_output_idx = self.trainer.datamodule.output_input_idx, self.trainer.datamodule.input_output_idx

    num_forecast_steps = num_forecast_steps or total_output_len

    max_input_window_idx = np.max([idx.max().cpu() for idx in input_window_idx])
    max_output_window_idx = np.max([idx.max().cpu() for idx in output_window_idx])

    # Calculate forecast length
    forecast_len = np.max([1, max_output_window_idx - max_input_window_idx])

    if eval:

      dl = None
      if self.trainer.datamodule.test_data is not None:
        test_data = self.trainer.datamodule.test_data
        if not isinstance(test_data, list): test_data = [test_data]
        if data['id'] in [data_['id'] for data_ in test_data]:
          if not hasattr(self.trainer.datamodule,'test_dl'):
            self.trainer.datamodule.predicting = True
            self.trainer.datamodule.test_dataloader() ;
            self.trainer.datamodule.predicting = False
          dl = self.trainer.datamodule.test_dl
      if (self.trainer.datamodule.val_data is not None) & (dl is None):
        val_data = self.trainer.datamodule.val_data
        if not isinstance(val_data, list): val_data = [val_data]
        if data['id'] in [data_['id'] for data_ in val_data]:
          dl = self.trainer.datamodule.val_dl
      if dl is None:
        shuffle_train_original = self.trainer.datamodule.shuffle_train
        if shuffle_train_original == True:
          self.trainer.datamodule.shuffle_train = False
          self.predicting = False
          self.trainer.datamodule.train_dataloader()
          self.predicting = True

        dl = self.trainer.datamodule.train_dl
        self.trainer.datamodule.shuffle_train = shuffle_train_original

      input, steps, ids = [], [], []
      for batch in dl.dl:
        input.append(batch[0][:batch[3]])
        steps.append(batch[2][:batch[3]])
        ids.append(batch[4])

      input = torch.cat(input, 0)
      steps = torch.cat(steps, 0)
      ids = np.concatenate(ids).tolist()

      id_idx = torch.tensor([id_idx for id_idx,id_ in enumerate(ids) if id_ == id]).to(device = input.device, dtype = torch.long)

      input = input[id_idx].reshape(len(id_idx), input.shape[1], input.shape[2])
      steps = steps[id_idx].reshape(len(id_idx), steps.shape[1])
      ids = [ids[id_idx_item] for id_idx_item in id_idx.tolist()]

      min_step, max_step = [], []
      if steps is not None:
        min_step, max_step = steps.min().item(), steps.max().item()

    else:

      input = torch.cat([data[name] for name in input_names], -1)[-total_input_len:].reshape(1, total_input_len, total_input_size)
      steps = data['step'][-total_input_len:].reshape(1, total_input_len)
      steps  = torch.cat((steps, steps.max()+torch.arange(1,total_window_size-total_input_len+1).reshape(1,-1).to(steps)), 1)
      num_samples = 1

    num_samples = input.shape[0]

    with torch.no_grad():

      # Initialize forecast tensors
      forecast = torch.empty((num_samples, 0, total_output_size)).to(device = self.model.device,
                                                                     dtype = self.model.dtype)
      forecast_steps = torch.empty((num_samples, 0)).to(device = self.model.device,
                                                        dtype = torch.long)

      # Generate forecast steps
      while forecast.shape[1] < num_forecast_steps: # (total_output_len + num_forecast_steps):
        # Generate prediction for the next forecast step
        prediction, hiddens = self.forward(input = input,
                                           steps = steps,
                                           hiddens = hiddens,
                                           input_window_idx = input_window_idx,
                                           output_window_idx = output_window_idx,
                                           output_mask = output_mask,
                                           output_input_idx = output_input_idx,
                                           input_output_idx = input_output_idx)

        # Create input for the next forecast step
        input_ = torch.zeros((num_samples, prediction.shape[1], total_input_size)).to(input)
        if len(output_input_idx) > 0:
          input_[:, :, output_input_idx] = prediction[:, -prediction.shape[1]:, input_output_idx]

        # Concatenate input for the next forecast step
        input = torch.cat((input[:, prediction.shape[1]:], input_), 1)

        # Append prediction to forecast
        forecast = torch.cat((forecast, prediction), 1)
        if steps is not None:
          forecast_steps = torch.cat((forecast_steps, steps[:, -prediction.shape[1]:]), 1)
          steps += prediction.shape[1]

      # Extract the relevant portion of the forecast
      forecast, forecast_steps = forecast[:, -num_forecast_steps:], forecast_steps[:, -num_forecast_steps:]

    forecast_target = None

    # Handle forecast target if forecast is in evaluation mode
    if eval:

      target = torch.cat([data[name] for name in output_names], -1)
      time = data[time_name]
      start_step = 0 # self.trainer.datamodule.start_step*self.trainer.datamodule.pad_data

      idx = (forecast_steps <= max_step).all(dim=1)

      forecast = forecast[idx] # .reshape(len(idx), -1, total_output_size)
      forecast_steps = forecast_steps[idx] # .reshape(len(idx), -1)

      if forecast.shape[0] <= 1:
        forecast = forecast.unsqueeze(0)
        forecast_steps = forecast_steps.unsqueeze(0)

      forecast_target = target[forecast_steps - start_step] # target[idx].reshape(len(idx), -1, total_output_size) #

      # plt.plot(forecast[0].cpu())
      # plt.plot(forecast_target[0].cpu())

      # Convert steps to forecasted time
      forecast_time = []
      for steps_ in forecast_steps:
        forecast_time.append(time.iloc[(steps_ - start_step).cpu().numpy()] if isinstance(time, pd.Series) else time[(steps_ - start_step).cpu().numpy()])

    else:

      start_time = data[time_name].max() + self.trainer.datamodule.dt

      forecast_time = pd.Series([start_time + n * self.trainer.datamodule.dt for n in range(num_forecast_steps)])
      if data[time_name].dt.tz is not None:
        forecast_time = forecast_time.tz_localize(data[time_name].dt.tz)

    # Reduce forecast if required
    inverted = False
    if invert:
      inverted = True
      j = 0
      for i, name in enumerate(output_names):
        for sample_idx in range(forecast.shape[0]):
          if output_feature_names is not None:
            f = 0
            for k, feature_name in enumerate(output_feature_names):
                if forecast_target is not None:
                  forecast_target[sample_idx, :, f:(f+output_feature_size[k])] = transforms[feature_name].inverse_transform(forecast_target[sample_idx, :, f:(f+output_feature_size[k])])
                forecast[sample_idx, :, f:(f+output_feature_size[k])] = transforms[feature_name].inverse_transform(forecast[sample_idx, :, f:(f+output_feature_size[k])])
          else:
            if forecast_target is not None:
              forecast_target[sample_idx, :, j:(j+self.model.output_size[i])] = transforms[name].inverse_transform(forecast_target[sample_idx, :, j:(j+self.model.output_size[i])])
            forecast[sample_idx, :, j:(j+self.model.output_size[i])] = transforms[name].inverse_transform(forecast[sample_idx, :, j:(j+self.model.output_size[i])])

    if not eval:
      forecast, forecast_time = forecast[0], forecast_time

      self.forecast_data = {'id': id,
                            'inverted': inverted,
                            time_name: forecast_time}

      j = 0
      for i, output_name in enumerate(output_names):
        self.forecast_data[f"{output_name}"] = forecast[:, j:(j+output_size[i])]
        j += output_size[i]
    else:
      return forecast, forecast_time, forecast_target
  ##

  ##
  def plot_forecast(self,
                    figsize = None,
                    window_len = 10):

    id = self.forecast_data['id']

    data, transforms = self.trainer.datamodule.data, self.trainer.datamodule.transforms
    if not isinstance(data, list):
      data = [data]
      transforms = [transforms]

    idx = [idx for idx,data_ in enumerate(data) if data_['id'] == id][0]
    data, transforms = data[idx], transforms[idx]

    time_name = self.trainer.datamodule.time_name
    time_unit = self.trainer.datamodule.time_unit
    input_names = self.trainer.datamodule.input_names
    output_names = self.trainer.datamodule.output_names
    output_feature_names = self.trainer.datamodule.output_feature_names
    output_feature_size = self.trainer.datamodule.output_feature_size
    num_outputs = len(output_names)

    input_window_idx = self.trainer.datamodule.train_input_window_idx
    output_window_idx = self.trainer.datamodule.train_output_window_idx
    total_input_len = len(torch.cat(input_window_idx).unique())
    total_output_len = len(torch.cat(output_window_idx).unique())

    forecast_time = self.forecast_data[time_name]

    figsize = figsize or (10, 5*num_outputs)
    fig, ax = plt.subplots(num_outputs, 1, figsize = figsize)

    time = data[time_name][-window_len:] # pd.concat([data[time_name][-window_len:], forecast_time])

    for i in range(num_outputs):
      ax_i = ax[i] if num_outputs > 1 else ax

      if self.forecast_data['inverted']:
        output_i = transforms[output_names[i]].inverse_transform(data[output_names[i]][-window_len:])
      else:
        output_i = data[output_names[i]][-window_len:]

      # output_i = torch.cat((output_i, self.forecast_data[output_names[i]]), 0)

      ax_i.plot(time, output_i.cpu(), '-*k', label = output_names[i])
      ax_i.plot(forecast_time, self.forecast_data[output_names[i]].cpu(), '-*b', label = f"{output_names[i]} forecast")
      ax_i.grid()
      ax_i.legend()

      # ax_i.axvspan(forecast_time.values.min(), forecast_time.values.max(),
      #              facecolor='black', alpha=0.1, label = 'Forecast')

      # Customize x-axis based on time unit
      if hasattr(time, "dt"):
        if time_unit == "S":
          ax_i.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
          ax_i.xaxis.set_major_locator(mdates.SecondLocator(interval=int(self.trainer.datamodule.dt.seconds)))
        elif time_unit == "M":
          ax_i.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
          ax_i.xaxis.set_major_locator(mdates.MinuteLocator(interval=int(self.trainer.datamodule.dt.seconds/60)))
        elif time_unit == "H":
          ax_i.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
          ax_i.xaxis.set_major_locator(mdates.HourLocator(interval=int(self.trainer.datamodule.dt.seconds/3600)))
        elif time_unit == "d":
          ax_i.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
          ax_i.xaxis.set_major_locator(mdates.DayLocator(interval=int(self.trainer.datamodule.dt.seconds/(3600*24))))

      ax_i.legend(loc = 'upper left', bbox_to_anchor = (1.02, 1), ncol = 1)
      ax_i.set_xlabel(f"Time [{time_unit}]")
      ax_i.set_ylabel(output_names[i])
      ax_i.set_title(f"ID {id}")

    fig.tight_layout()
  ##

  ##
  def backtest(self,
               num_forecast_steps = None,
               ids = None,
               stride = 1,
               hiddens = None, invert = True):

    """
    Backtest the model's performance.

    Args:
      ids (list or None): List of IDs to perform backtesting on. If None, backtesting is performed on all available IDs.
      num_forecast_steps (int): Number of forecast steps to predict.
      stride (int): Step size for sampling the input data.
      hiddens (list or None): List of hidden states.
      invert (bool): Whether to invert the transformation during backtesting.

    Returns:
      None
    """

    if ids is None:
      if self.trainer.datamodule.test_data is not None:
        data = self.trainer.datamodule.test_data
      elif self.trainer.datamodule.val_data is not None:
        data = self.trainer.datamodule.val_data
      else:
        data = self.trainer.datamodule.train_data

      if not isinstance(data, list): data = [data]

      ids = [data_['id'] for data_ in data]

    if self.accelerator == 'gpu':
      self.model.to('cuda')

    if not isinstance(hiddens, list):
      hiddens = [hiddens]

    time_name = self.trainer.datamodule.time_name
    output_names = self.trainer.datamodule.output_names
    output_feature_names = self.trainer.datamodule.output_feature_names
    output_feature_size = self.trainer.datamodule.output_feature_size

    self.backtest_data = []

    for id in ids:

      self.backtest_data.append({'id': id})

      # id_idx = torch.tensor(np.where(ids == id)[0]).to(device=input.device, dtype=torch.long)

      hiddens_id = hiddens[-1]
      forecast_id, forecast_time_id, forecast_target_id = self.forecast(num_forecast_steps,
                                                                        id = id,
                                                                        hiddens = hiddens_id,
                                                                        invert = invert,
                                                                        eval = True)

      forecast_id = forecast_id[::-stride][::-1]
      forecast_time_id = forecast_time_id[::-stride][::-1]
      forecast_target_id = forecast_target_id[::-stride][::-1]

      self.backtest_data[-1][time_name] = forecast_time_id

      j = 0
      for i, name in enumerate(self.trainer.datamodule.output_names):

        self.backtest_data[-1][f"{name}_target"] = forecast_target_id[..., j:(j+self.trainer.datamodule.output_size[i])]
        self.backtest_data[-1][f"{name}_prediction"] = forecast_id[..., j:(j+self.trainer.datamodule.output_size[i])]

        # Compute global loss for each output
        metric_name_i = None
        if isinstance(self.loss_fn, list):
          loss_name_i = self.loss_fn[i].name
          metric_name_i = self.metric_fn[i].name if self.metric_fn[i] is not None else None
        else:
          loss_name_i = self.loss_fn.name
          metric_name_i = self.metric_fn.name if self.metric_fn is not None else None

        self.backtest_data[-1][f"{name}_{loss_name_i}"] = Criterion(loss_name_i,
                                                                    dims=1)(self.backtest_data[-1][f"{name}_prediction"],
                                                                            self.backtest_data[-1][f"{name}_target"])

        if metric_name_i is not None:
          self.backtest_data[-1][f"{name}_{metric_name_i}"] = Criterion(metric_name_i,
                                                                        dims=1)(self.backtest_data[-1][f"{name}_prediction"],
                                                                                self.backtest_data[-1][f"{name}_target"])

        j += self.trainer.datamodule.output_size[i]

    if self.trainer.datamodule.num_datasets == 1:
      self.backtest_data = self.backtest_data[0]
  ##

  ##
  def plot_backtest(self, id = None, figsize = None, num_backtests = 1):

    """
    Plot the backtest results for a specific ID.

    Args:
        id (str): The ID for which the backtest results are plotted.

    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure.
    """

    self.num_backtests = num_backtests

    # Ensure backtest_data is a list
    backtest_data = self.backtest_data
    if not isinstance(backtest_data, list): backtest_data = [backtest_data]

    if id is None:
      id = backtest_data[0]['id']

    # Get necessary data and parameter values
    output_names = self.trainer.datamodule.output_names
    time_name = self.trainer.datamodule.time_name

    # Find the index of the requested ID in the backtest_data list
    data_idx = [idx for idx, data in enumerate(backtest_data) if data['id'] == id][0]

    # Extract the backtest data for the specified ID
    backtest_data_idx = backtest_data[data_idx]

    figsize = figsize or (7 * num_backtests, 5 * self.model.num_outputs)

    # Create a matplotlib figure with subplots
    fig, ax = plt.subplots(self.trainer.datamodule.num_outputs,
                           num_backtests,
                           figsize=figsize)

    # Extract forecast time data
    forecast_time = backtest_data_idx[time_name]

    # Iterate through each output and backtest step
    for i in range(self.trainer.datamodule.num_outputs):
      for j, (time_i, target_i, prediction_i) in enumerate(zip(forecast_time[-num_backtests:],
                                                               backtest_data_idx[f"{output_names[i]}_target"][-num_backtests:].cpu(),
                                                               backtest_data_idx[f"{output_names[i]}_prediction"][-num_backtests:].cpu())):
          ax_ji = (ax[i, j]
                  if self.model.num_outputs > 1
                  else ax[j]
                  if num_backtests > 1
                  else ax)

          # Plot time series data
          ax_ji.plot(time_i, target_i, "k", label="Target")
          ax_ji.plot(time_i, prediction_i, "r", label="Prediction")

          # Customize x-axis based on time unit
          if hasattr(time_i, "dt"):
            if self.trainer.datamodule.time_unit == "S":
              ax_ji.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
              ax_ji.xaxis.set_major_locator(mdates.SecondLocator(interval=int(self.trainer.datamodule.dt.seconds)))
            elif self.trainer.datamodule.time_unit == "M":
              ax_ji.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M:%S"))
              ax_ji.xaxis.set_major_locator(mdates.MinuteLocator(interval=int(self.trainer.datamodule.dt.seconds/60)))
            elif self.trainer.datamodule.time_unit == "H":
              ax_ji.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))
              ax_ji.xaxis.set_major_locator(mdates.HourLocator(interval=int(self.trainer.datamodule.dt.seconds/3600)))
            elif self.trainer.datamodule.time_unit == "d":
              ax_ji.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
              ax_ji.xaxis.set_major_locator(mdates.DayLocator(interval=int(self.trainer.datamodule.dt.seconds/(3600*24))))

          loss_ji = backtest_data_idx[f"{output_names[i]}_{self.loss_fn.name}"][-num_backtests+j].cpu().item()

          metric_ji = None
          if self.metric_fn is not None:
            metric_ji = backtest_data_idx[f"{output_names[i]}_{self.metric_fn.name}"][-num_backtests+j].cpu().item()

          # Set grid and labels
          ax_ji.grid()
          ax_ji.set_title(f"ID {id}: {self.loss_fn.name.upper()} = {np.round(loss_ji, 4)}, {self.metric_fn.name.upper()} = {np.round(metric_ji, 4)}" if metric_ji is not None
                                     else f"ID {id}: {self.loss_fn.name.upper()} = {np.round(loss_ji, 4)}")
          ax_ji.set_xlabel("Time")
          if j == 0: ax_ji.set_ylabel(output_names[i])
          ax_ji.legend()

    plt.tight_layout()

    # return fig
  ##

  ##
  def generate_reduced_output(self, output, output_steps, reduction='mean', transforms=None):
    """
    Generate reduced output based on specified reduction method and optional transforms.

    Args:
        output (torch.Tensor): The original output tensor.
        output_steps (torch.Tensor): The steps associated with the output.
        reduction (str, optional): The reduction method to use ('mean' or 'median').
            Defaults to 'mean'.
        transforms (dict, optional): The data transforms to apply. Defaults to None.

    Returns:
        torch.Tensor: The reduced output tensor.
        torch.Tensor: The unique output steps corresponding to the reduced output.
    """
    # Get unique output steps and remove any -1 values
    unique_output_steps = output_steps.unique()
    unique_output_steps = unique_output_steps[unique_output_steps != -1]

    # Create a tensor to store the reduced output
    output_reduced = torch.zeros((len(unique_output_steps), sum(self.model.output_size))).to(output)

    output_names = self.trainer.datamodule.output_feature_names or self.trainer.datamodule.output_names
    output_feature_names = self.trainer.datamodule.output_feature_names
    output_feature_size = self.trainer.datamodule.output_feature_size

    k = -1
    for step in unique_output_steps:
      k += 1

      # Find the indices of the current step in the output_steps tensor
      batch_step_idx = torch.where(output_steps == step)
      num_step_output = len(batch_step_idx[0])

      j = 0
      for i in range(self.model.num_outputs):

        # Extract the output for the current output index
        output_i = output[:, :, j:(j + self.model.output_size[i])]
        output_reduced_i = []

        step_output_i = []
        for batch_idx, step_idx in zip(*batch_step_idx[:2]):
            step_output_i.append(output_i[batch_idx, step_idx, :].reshape(1, 1, -1))

        if len(step_output_i) > 0:
            step_output_i = torch.cat(step_output_i, 0)

            # Reduce the step outputs based on the specified reduction method
            step_output_reduced_i = (step_output_i.median(0)[0] if reduction == 'median' else
                                     step_output_i.mean(0)).reshape(-1, self.model.output_size[i])

            # Assign the reduced output to the output_reduced tensor
            output_reduced[k, j:(j + self.model.output_size[i])] = step_output_reduced_i.squeeze(0)

        j += self.model.output_size[i]

    # Optionally invert the reduced output using data scalers
    if transforms is not None:
      j = 0
      for i, feature_name in enumerate(self.trainer.datamodule.output_feature_names):
        output_reduced[:, j:(j + output_feature_size[i])] = transforms[feature_name].inverse_transform(output_reduced[:, j:(j + output_feature_size[i])])
        j += output_feature_size[i]

      # j = 0
      # for i, name in enumerate(self.trainer.datamodule.output_names):
      #     if output_feature_names is not None:
      #         f = 0
      #         for k, feature_name in enumerate(output_feature_names):
      #           output_reduced[:, f:(f + output_feature_size[k])] = transforms[feature_name].inverse_transform(output_reduced[:, f:(f + output_feature_size[k])])
      #     else:
      #         output_reduced[:, j:(j + self.model.output_size[i])] = transforms[name].inverse_transform(output_reduced[:, j:(j + self.model.output_size[i])])

    # Return the reduced output and unique output steps
    return output_reduced, unique_output_steps

  def fit(self, datamodule, max_epochs=20, callbacks=[None]):
    """
    Fit the model using the specified datamodule and training configuration.

    Args:
        datamodule (pl.LightningDataModule): The data module for training.
        max_epochs (int, optional): The maximum number of epochs for training. Defaults to 20.
        callbacks (list, optional): List of callbacks to be used during training. Defaults to [None].
    """
    # Set predicting flag to False
    datamodule.predicting = False

    try:
      # Create a Trainer instance and fit the model
      self.trainer = pl.Trainer(max_epochs=max_epochs,
                                accelerator=self.accelerator,
                                callbacks=callbacks)
      self.trainer.fit(self, datamodule=datamodule)

    except KeyboardInterrupt:
      # Handle KeyboardInterrupt by restoring the model's state
      state_dict = self.model.state_dict()
      self.model.to(device=self.model.device, dtype=self.model.dtype)
      self.model.load_state_dict(state_dict)
