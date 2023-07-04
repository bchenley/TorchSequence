import pytorch_lightning as pl
import torch
import numpy as np
import time

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from src import Loss

class SequenceModule(pl.LightningModule):
  def __init__(self,
               model,
               opt, loss_fn, metric_fn = None,
               constrain = False, penalize = False,
               track = False,
               model_dir = None):

    super().__init__()

    self.automatic_optimization = False

    self.model = model

    self.opt, self.loss_fn, self.metric_fn = opt, loss_fn, metric_fn

    self.constrain, self.penalize = constrain, penalize

    input_size, output_size = self.model.input_size, self.model.output_size

    self.train_history, self.val_history = None, None
    self.current_val_epoch = 0

    self.train_step_loss = []
    self.val_step_loss = []
    self.test_step_loss = []

    self.hiddens = None

    self.track = track

    self.model_dir = model_dir

  def forward(self,
              input,
              hiddens = None,
              steps = None,
              target = None,
              output_window_idx = None,
              output_mask = None,
              output_input_idx = None, input_output_idx = None,
              encoder_output= None):

    output, hiddens = self.model.forward(input = input,
                                         steps = steps,
                                        hiddens = hiddens,
                                        target = target,
                                        output_window_idx = output_window_idx,
                                        output_mask = output_mask,
                                        output_input_idx = output_input_idx,
                                        input_output_idx = input_output_idx,
                                        encoder_output= encoder_output)

    return output, hiddens
  
  ## Configure optimizers
  def configure_optimizers(self):
    return self.opt
  ##
   
  ## train model
  def on_train_start(self):
    self.run_time = time.time()
   
  def training_step(self, batch, batch_idx):

    # constrain model if desired
    if self.constrain: self.model.constrain()
    #

    # unpack batch
    input_batch, output_batch, steps_batch, batch_size = batch
    #

    # keep the first `batch_size` batches of hiddens
    if self.hiddens is not None:
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
    output_batch = output_batch[:batch_size]
    steps_batch = steps_batch[:batch_size]
    #

    # perform forward pass to compute gradients
    output_pred_batch, self.hiddens = self.forward(input = input_batch,
                                                   steps = steps_batch,
                                                   hiddens = self.hiddens,
                                                   target = output_batch,
                                                   output_window_idx = self.trainer.datamodule.train_output_window_idx,
                                                   output_input_idx = self.trainer.datamodule.output_input_idx,
                                                   input_output_idx = self.trainer.datamodule.input_output_idx,
                                                   output_mask = self.trainer.datamodule.train_output_mask)
    #

    # get loss for each output
    loss = self.loss_fn(output_pred_batch*self.trainer.datamodule.train_output_mask,
                        output_batch*self.trainer.datamodule.train_output_mask)
    loss = torch.stack([l.sum() for l in loss.split(self.model.output_size, -1)], 0)
    #

    # add penalty loss if desired
    if self.penalize: loss += self.model.penalize()
    #

    self.opt.zero_grad()
    loss.sum().backward()
    self.opt.step()

    # store loss to be used later in `on_train_epoch_end`
    self.train_step_loss.append(loss)
    #

    return {"loss": loss}

  def on_train_batch_start(self, batch, batch_idx):
    if self.hiddens is not None:
      for i in range(self.model.num_inputs):
        if (self.model.base_type[i] in ['gru', 'lstm', 'lru']) & (self.hiddens[i] is not None):
          if self.model.base_type[i] == 'lstm':
            self.hiddens[i] = [s.detach() for s in self.hiddens[i]]
          else:
            self.hiddens[i] = self.hiddens[i].detach()

  def on_train_batch_end(self, outputs, batch, batch_idx):

    # reduced loss of current batch
    train_step_loss = outputs['loss'].detach()
    #

    # log and display sum of batch loss
    self.log('train_step_loss', train_step_loss.sum(), on_step = True, prog_bar = True)
    #

    if self.track:
      if self.train_history is None:
        self.current_train_step = 0
        self.train_history = {'steps': torch.empty((0, 1)).to(device = train_step_loss.device,
                                                              dtype = torch.long)}
        for i in range(self.model.num_outputs):
          loss_name_i = self.loss_fn.name + '_' + self.trainer.datamodule.output_names[i]
          self.train_history[loss_name_i] = torch.empty((0, 1)).to(train_step_loss)

        for name, param in self.model.named_parameters():
          if param.requires_grad == True:
            self.train_history[name] = torch.empty((0, param.numel())).to(param)

      else:
        self.train_history['steps'] = torch.cat((self.train_history['steps'],
                                                 torch.tensor(self.current_train_step).reshape(1, 1).to(train_step_loss)), 0)

        for i in range(self.trainer.datamodule.num_outputs):
          loss_name_i = self.loss_fn.name + '_' + self.trainer.datamodule.output_names[i]
          self.train_history[loss_name_i] = torch.cat((self.train_history[loss_name_i],
                                                       train_step_loss[i].cpu().reshape(1, 1).to(train_step_loss)), 0)

        for i,(name, param) in enumerate(self.model.named_parameters()):
          if param.requires_grad:
            self.train_history[name] = torch.cat((self.train_history[name],
                                                  param.clone().detach().cpu().reshape(1, -1).to(param)), 0)

      self.current_train_step += 1

  def on_train_epoch_start(self):
    self.hiddens = None
    self.train_step_loss = []

  def on_train_epoch_end(self):

    # epoch loss
    train_epoch_loss = torch.stack(self.train_step_loss).mean(0)
    #

    self.log('train_epoch_loss', train_epoch_loss.sum(), on_epoch = True, prog_bar = True)

    self.train_step_loss.clear()
  ## End of Training

  ## Validate Model
  def validation_step(self, batch, batch_idx):

    # unpack batch
    input_batch, output_batch, steps_batch, batch_size = batch
    #

    # keep the first `batch_size` batches of hiddens
    if self.hiddens is not None:

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
    output_batch = output_batch[:batch_size]
    steps_batch = steps_batch[:batch_size]
    #

    # perform forward pass to compute gradients
    output_pred_batch, self.hiddens = self.forward(input = input_batch,
                                                  steps = steps_batch,
                                                  hiddens = self.hiddens,
                                                  target = None,
                                                  output_window_idx = self.trainer.datamodule.val_output_window_idx,
                                                  output_input_idx = self.trainer.datamodule.output_input_idx,
                                                  input_output_idx = self.trainer.datamodule.input_output_idx,
                                                  output_mask = self.trainer.datamodule.val_output_mask)
    #

    # get loss for each output
    loss = self.loss_fn(output_pred_batch*self.trainer.datamodule.val_output_mask,
                        output_batch*self.trainer.datamodule.val_output_mask)
    loss = torch.stack([l.sum() for l in loss.split(self.model.output_size, -1)], 0)
    #

    self.val_step_loss.append(loss)

    {"loss": loss}

  def on_validation_epoch_end(self):
    # epoch loss
    val_epoch_loss = torch.stack(self.val_step_loss).mean(0)
    #

    self.log('val_epoch_loss', val_epoch_loss.sum(), on_step = False, on_epoch = True, prog_bar = True)

    if self.track:
      if self.val_history is None:
        self.val_history = {'epochs': torch.empty((0, 1)).to(device = val_epoch_loss.device,
                                                             dtype = torch.long)}
        for i in range(self.trainer.datamodule.num_outputs):
          self.val_history[self.loss_fn.name + '_' + self.trainer.datamodule.output_names[i]] = torch.empty((0, 1)).to(val_epoch_loss)

      else:
        self.val_history['epochs'] = torch.cat((self.val_history['epochs'],
                                              torch.tensor(self.current_val_epoch).reshape(1, 1).to(val_epoch_loss)), 0)

        for i in range(self.trainer.datamodule.num_outputs):
          loss_name_i = self.loss_fn.name + '_' + self.trainer.datamodule.output_names[i]
          self.val_history[loss_name_i] = torch.cat((self.val_history[loss_name_i],
                                                    val_epoch_loss[i].cpu().reshape(1, 1).to(val_epoch_loss)), 0)
          
      self.current_val_epoch += 1

    self.val_step_loss.clear()

    
  ## End of validation

  ## Test Model
  def test_step(self, batch, batch_idx):

    # unpack batch
    input_batch, output_batch, steps_batch, batch_size = batch
    #

    # keep the first `batch_size` batches of hiddens
    if self.hiddens is not None:
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
    output_batch = output_batch[:batch_size]
    steps_batch = steps_batch[:batch_size]
    #

    # perform forward pass to compute gradients
    output_pred_batch, self.hiddens = self.forward(input = input_batch,
                                                  steps = steps_batch,
                                                  hiddens = self.hiddens,
                                                  target = None,
                                                  output_window_idx = self.trainer.datamodule.test_output_window_idx,
                                                  output_input_idx = self.trainer.datamodule.output_input_idx,
                                                  input_output_idx = self.trainer.datamodule.input_output_idx,
                                                  output_mask = self.trainer.datamodule.test_output_mask)
    #

    # get loss for each output
    loss = self.loss_fn(output_pred_batch*self.trainer.datamodule.test_output_mask,
                        output_batch*self.trainer.datamodule.test_output_mask)
    loss = torch.stack([l.sum() for l in loss.split(self.model.output_size, -1)], 0)
    #

    self.test_step_loss.append(loss)

    {"loss": loss}

  def on_test_epoch_end(self):
    # epoch loss
    test_epoch_loss = torch.stack(self.test_step_loss).mean(0)
    self.test_step_loss.clear()
    #

    self.log('test_epoch_loss', test_epoch_loss.sum(), on_epoch = True, prog_bar = True)
  ## End of Testing

  ## plot history
  def plot_history(self, history = None, plot_train_history_by = 'epochs'):

    history = [self.loss_fn.name] if history is None else history

    if plot_train_history_by == 'epochs':
      num_batches = len(self.trainer.datamodule.train_dl.dl)
      train_history_epoch = {'epochs': torch.arange(len(self.train_history['steps'])//num_batches).to(dtype = torch.long)}
      num_epochs = len(train_history_epoch['epochs'])
      for key in self.train_history.keys():
        if key != 'steps':
          batch_param = []
          for batch in self.train_history[key].split(num_batches, 0):
            batch_param.append(batch.mean(0, keepdim = True))
          batch_param = torch.cat(batch_param, 0)
          train_history_epoch[key] = batch_param[:num_epochs]

      train_history = train_history_epoch

      x_label = 'epochs'

    else:
      x_label = 'steps'
      train_history = self.train_history

    num_params = len(history)
    fig = plt.figure(figsize = (5, 5*num_params))
    ax_i = 0
    for param in history:
      ax_i += 1
      ax = fig.add_subplot(num_params, 1, ax_i)
      ax.plot(train_history[x_label], train_history[param], label = 'Train')
      if (self.val_history is not None) & (param in self.val_history) & (x_label == 'epochs'):
        N = np.min([self.val_history[x_label].shape[0], self.val_history[param].shape[0]])

        if self.loss_fn.name in param:
          metric = self.val_history[param][:N]
        elif self.metric_fn.name is not None:
          if self.metric_fn.name in param:
            metric = self.val_history[param][:N]
        
        ax.plot(self.val_history[x_label][:N], metric, label = 'Val')
      ax.set_title(param)
      ax.set_ylabel(param)
      ax.legend()
    plt.grid()
  ##

  ## Prediction
  def predict_step(self, batch, batch_idx):

    # unpack batch
    input_batch, output_batch, steps_batch, batch_size = batch
    #

    # keep the first `batch_size` batches of hiddens
    if self.hiddens is not None:
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
    output_batch = output_batch[:batch_size]
    steps_batch = steps_batch[:batch_size]
    #

    output_len = output_batch.shape[1]

    # perform forward pass to compute gradients
    output_pred_batch, self.hiddens = self.forward(input = input_batch,
                                                   steps = steps_batch,
                                                   hiddens = self.hiddens,
                                                   target = None,
                                                   output_window_idx = self.predict_output_window_idx,
                                                   output_input_idx = self.trainer.datamodule.output_input_idx,
                                                   input_output_idx = self.trainer.datamodule.input_output_idx,
                                                   output_mask = self.predict_output_mask)
    #

    # get loss for each output
    step_loss = self.loss_fn(output_pred_batch*self.predict_output_mask,
                             output_batch*self.predict_output_mask)
    step_loss = torch.stack([l.sum() for l in step_loss.split(self.model.input_size, -1)], 0)
    #

    output_steps_batch = steps_batch[:, -output_len:]

    return output_batch, output_pred_batch, output_steps_batch # , baseline_pred_batch

  def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
    self.step_target.append(outputs[0])
    self.output_pred_batch.append(outputs[1])
    self.output_steps_batch.append(outputs[2])
    # self.step_baseline_pred.append(outputs[3])

  def on_predict_epoch_end(self):
    self.target = torch.cat(self.step_target, 0)
    self.prediction = torch.cat(self.output_pred_batch, 0)
    self.output_steps = torch.cat(self.output_steps_batch, 0)
    # self.baseline_prediction = torch.cat(self.step_baseline_pred, 0)

    self.step_target.clear()
    self.output_pred_batch.clear()
    self.output_steps_batch.clear()
    # self.step_baseline_pred.clear()

  def on_predict_epoch_start(self):
    self.output_pred_batch, self.step_target = [], []
    self.output_steps_batch = []
    # self.step_baseline_pred = []

  def predict(self,
              reduction = 'mean',
              baseline_model = None):

    self.baseline_model = baseline_model

    self.trainer.datamodule.predicting = True

    self.trainer.enable_progress_bar = False

    start_step = self.trainer.datamodule.start_step

    with torch.no_grad():

      ## Predict training data
      self.hiddens = None
      self.predict_output_mask = self.trainer.datamodule.train_output_mask
      self.predict_output_window_idx = self.trainer.datamodule.train_output_window_idx

      self.trainer.predict(self, self.trainer.datamodule.train_dl.dl)

      if self.trainer.datamodule.pad_data:
        self.prediction, self.target, self.output_steps = self.prediction[start_step:], self.target[start_step:], self.output_steps[start_step:]

      train_prediction, train_output_steps = self.generate_reduced_output(self.prediction, self.output_steps,
                                                                          reduction = reduction, transforms=self.trainer.datamodule.transforms)

      train_target, _ = self.generate_reduced_output(self.target, self.output_steps,
                                                     reduction = reduction, transforms=self.trainer.datamodule.transforms)

      # train_loss = self.loss_fn(train_prediction.unsqueeze(0),
      #                           train_target.unsqueeze(0))

      # train_loss = torch.stack([l.sum() for l in train_loss.split(self.model.input_size, -1)], 0)

      train_time = self.trainer.datamodule.train_data[self.trainer.datamodule.time_name][start_step:]
      
      train_baseline_pred, train_baseline_loss = None, None
      if self.baseline_model is not None:
        train_baseline_pred = self.baseline_model(train_target)
        # train_baseline_loss = self.loss_fn(train_baseline_pred.unsqueeze(0),
        #                                    train_target.unsqueeze(0))
      ##

      # Predict validation data
      val_prediction, val_target, val_time, val_loss, val_baseline_pred, val_baseline_loss = None, None, None, None, None, None
      if len(self.trainer.datamodule.val_dl.dl) > 0:
        self.predict_output_mask = self.trainer.datamodule.val_output_mask
        self.predict_output_window_idx = self.trainer.datamodule.val_output_window_idx

        self.trainer.predict(self, self.trainer.datamodule.val_dl.dl) ;
        
        val_prediction, val_output_steps = self.generate_reduced_output(self.prediction, self.output_steps,
                                                                        reduction = reduction, transforms=self.trainer.datamodule.transforms)
        
        val_target, _ = self.generate_reduced_output(self.target, self.output_steps,
                                                     reduction = reduction, transforms=self.trainer.datamodule.transforms)

        # val_loss = self.loss_fn(val_prediction.unsqueeze(0),
        #                         val_target.unsqueeze(0))
        # val_loss = torch.stack([l.sum() for l in val_loss.split(self.model.input_size, -1)], 0)

        val_time = self.trainer.datamodule.val_data[self.trainer.datamodule.time_name]

        if not self.trainer.datamodule.pad_data:
          val_time = val_time[start_step:]
                  
        val_baseline_pred, val_baseline_loss = None, None
        if self.baseline_model is not None:
          val_baseline_pred = self.baseline_model(val_target)
          # val_baseline_loss = self.loss_fn(val_baseline_pred.unsqueeze(0),
          #                                  val_target.unsqueeze(0))
      #

      # Predict testing data
      if not hasattr(self.trainer.datamodule, 'test_dl'):
        self.trainer.datamodule.test_dataloader()
      test_prediction, test_target, test_time, test_loss, test_baseline_pred, test_baseline_loss = None, None, None, None, None, None
      if len(self.trainer.datamodule.test_dl.dl) > 0:
        self.predict_output_mask = self.trainer.datamodule.test_output_mask
        self.predict_output_window_idx = self.trainer.datamodule.test_output_window_idx

        self.trainer.predict(self, self.trainer.datamodule.test_dl.dl) ;

        test_prediction, test_output_steps = self.generate_reduced_output(self.prediction, self.output_steps,
                                                                          reduction = reduction, transforms=self.trainer.datamodule.transforms)

        test_target, _ = self.generate_reduced_output(self.target, self.output_steps,
                                                      reduction = reduction, transforms=self.trainer.datamodule.transforms)

        # test_loss = self.loss_fn(test_prediction.unsqueeze(0),
        #                         test_target.unsqueeze(0))
        # test_loss = torch.stack([l.sum() for l in test_loss.split(self.model.input_size, -1)], 0)

        test_time = self.trainer.datamodule.test_data[self.trainer.datamodule.time_name]

        if not self.trainer.datamodule.pad_data:
          test_time = test_time[start_step:]
                  
        test_baseline_pred, test_baseline_loss = None, None
        if self.baseline_model is not None:
          test_baseline_pred = self.baseline_model(test_target)
          # test_baseline_loss = self.loss_fn(test_baseline_pred.unsqueeze(0),
          #                                   test_target.unsqueeze(0))
      #

    train_prediction_data, val_prediction_data, test_prediction_data = {self.trainer.datamodule.time_name: train_time}, None, None

    if val_prediction is not None: val_prediction_data = {self.trainer.datamodule.time_name: val_time}
    if test_prediction is not None: test_prediction_data = {self.trainer.datamodule.time_name: test_time}
                
    j = 0
    for i,output_name in enumerate(self.trainer.datamodule.output_names):

      # train
      train_target_i = train_target[:, j:(j+self.trainer.datamodule.output_size[i])]
      train_prediction_i = train_prediction[:, j:(j+self.trainer.datamodule.output_size[i])]
      
      train_prediction_data[f"{output_name}_actual"] = train_target_i
      train_prediction_data[f"{output_name}_prediction"] = train_prediction_i

      train_loss_i = Loss(self.loss_fn.name,
                          dims=(0,1))(train_prediction_i.unsqueeze(0), train_target_i.unsqueeze(0))
      train_prediction_data[f"{output_name}_{self.loss_fn.name}"] = train_loss_i

      if self.metric_fn is not None:
        train_metric_i = Loss(self.metric_fn.name,
                            dims=(0,1))(train_prediction_i.unsqueeze(0), train_target_i.unsqueeze(0))
        train_prediction_data[f"{output_name}_{self.metric_fn.name}"] = train_metric_i

      train_baseline_pred_i, train_baseline_loss_i, train_baseline_metric_i = None, None, None
      if train_baseline_pred is not None:
        train_baseline_pred_i = train_baseline_pred[:, j:(j+self.trainer.datamodule.output_size[i])]

        train_baseline_loss_i = Loss(self.loss_fn.name,
                                     dims=(0,1))(train_baseline_pred_i.unsqueeze(0), train_target_i.unsqueeze(0))

        if self.metric_fn is not None:
          train_baseline_metric_i = Loss(self.metric_fn.name,
                                         dims=(0,1))(train_baseline_pred_i.unsqueeze(0), train_target_i.unsqueeze(0))

      train_prediction_data[f"{output_name}_baseline_prediction"] = train_baseline_pred_i
      train_prediction_data[f"{output_name}_baseline_{self.loss_fn.name}"] = train_baseline_loss_i

      if self.metric_fn is not None:
        train_prediction_data[f"{output_name}_baseline_{self.metric_fn.name}"] = train_baseline_metric_i
      #

      # val
      if val_prediction is not None:
        val_prediction_data[output_name] = {}

        val_target_i = val_target[:, j:(j+self.trainer.datamodule.output_size[i])]
        val_prediction_i = val_prediction[:, j:(j+self.trainer.datamodule.output_size[i])]

        val_prediction_data[f"{output_name}_actual"] = val_target_i
        val_prediction_data[f"{output_name}_prediction"] = val_prediction_i

        val_loss_i = Loss(self.loss_fn.name,
                            dims=(0,1))(val_prediction_i.unsqueeze(0), val_target_i.unsqueeze(0))
        val_prediction_data[f"{output_name}_{self.loss_fn.name}"] = val_loss_i

        if self.metric_fn is not None:
          val_metric_i = Loss(self.metric_fn.name,
                              dims=(0,1))(val_prediction_i.unsqueeze(0), val_target_i.unsqueeze(0))
          val_prediction_data[f"{output_name}_{self.metric_fn.name}"] = val_metric_i

        val_baseline_pred_i, val_baseline_loss_i, val_baseline_metric_i = None, None, None
        if val_baseline_pred is not None:
          val_baseline_pred_i = val_baseline_pred[:, j:(j+self.trainer.datamodule.output_size[i])]

          val_baseline_loss_i = Loss(self.loss_fn.name,
                              dims=(0,1))(val_baseline_pred_i.unsqueeze(0), val_target_i.unsqueeze(0))

          if self.metric_fn is not None:
            val_baseline_metric_i = Loss(self.metric_fn.name,
                                         dims=(0,1))(val_baseline_pred_i.unsqueeze(0), val_target_i.unsqueeze(0))

        val_prediction_data[f"{output_name}_baseline_prediction"] = val_baseline_pred_i
        val_prediction_data[f"{output_name}_baseline_{self.loss_fn.name}"] = val_baseline_loss_i

        if self.metric_fn is not None:
          val_prediction_data[f"{output_name}_baseline_{self.metric_fn.name}"] = val_baseline_metric_i
      #

      # test
      if test_prediction is not None:
        test_prediction_data[output_name] = {}

        test_target_i = test_target[:, j:(j+self.trainer.datamodule.output_size[i])]
        test_prediction_i = test_prediction[:, j:(j+self.trainer.datamodule.output_size[i])]

        test_prediction_data[f"{output_name}_actual"] = test_target_i
        test_prediction_data[f"{output_name}_prediction"] = test_prediction_i

        test_loss_i = Loss(self.loss_fn.name,
                           dims=(0,1))(test_prediction_i.unsqueeze(0), test_target_i.unsqueeze(0))
        test_prediction_data[f"{output_name}_{self.loss_fn.name}"] = test_loss_i

        if self.metric_fn is not None:
          test_metric_i = Loss(self.metric_fn.name,
                              dims=(0,1))(test_prediction_i.unsqueeze(0), test_target_i.unsqueeze(0))
          test_prediction_data[f"{output_name}_{self.metric_fn.name}"] = test_metric_i

        test_baseline_pred_i, test_baseline_loss_i, test_baseline_metric_i = None, None, None
        if test_baseline_pred is not None:
          test_baseline_pred_i = test_baseline_pred[:, j:(j+self.trainer.datamodule.output_size[i])]

          test_baseline_loss_i = Loss(self.loss_fn.name,
                                      dims=(0,1))(test_baseline_pred_i.unsqueeze(0), test_target_i.unsqueeze(0))

          if self.metric_fn is not None:
            test_baseline_metric_i = Loss(self.metric_fn.name,
                                          dims=(0,1))(test_baseline_pred_i.unsqueeze(0), test_target_i.unsqueeze(0))

        test_prediction_data[f"{output_name}_baseline_prediction"] = test_baseline_pred_i
        test_prediction_data[f"{output_name}_baseline_{self.loss_fn.name}"] = test_baseline_loss_i

        if self.metric_fn is not None:
          test_prediction_data[f"{output_name}_baseline_{self.metric_fn.name}"] = test_baseline_metric_i
      #

      j += self.trainer.datamodule.output_size[i]

    self.train_prediction_data, self.val_prediction_data, self.test_prediction_data = train_prediction_data, val_prediction_data, test_prediction_data

    self.trainer.enable_progress_bar = True
    self.trainer.datamodule.predicting = False

  ##
  def plot_predictions(self,
                       output_feature_units = None,
                       include_baseline = False):

    time_name = self.trainer.datamodule.time_name
    output_names = self.trainer.datamodule.output_names
    feature_names = self.trainer.datamodule.feature_names
    num_outputs = len(output_names)
    output_size = self.trainer.datamodule.output_size
    max_output_size = np.max(output_size)

    start_step = self.trainer.datamodule.start_step

    rows, cols = max_output_size, num_outputs
    fig, ax = plt.subplots(rows, cols, figsize = (10*num_outputs, 5*max_output_size))

    train_time = self.train_prediction_data[time_name]
    val_time = self.val_prediction_data[time_name] if self.val_prediction_data is not None else None
    test_time = self.test_prediction_data[time_name] if self.test_prediction_data is not None else None

    for i,output_name in enumerate(output_names):

      try:
        ax_i = ax[i, :]
        [ax_j.axis("off") for ax_j in ax_i]
      except:
        pass

      for f in range(output_size[i]):

        if (feature_names is not None):
          if any(output_name in name for name in feature_names) & (output_size[i] > 1):
            output_feature_name_if = feature_names[output_name][f]
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
          ax_if = ax[f,i]
        except:
          try:
            j = i if (cols>1) & (rows == 1) else f
            ax_if = ax[j]
          except:
            ax_if = ax

        train_target_if = self.train_prediction_data[f"{output_name}_actual"][:, f]
        train_prediction_if = self.train_prediction_data[f"{output_name}_prediction"][:, f]
        train_loss_if = np.round(self.train_prediction_data[f"{output_name}_{self.loss_fn.name}"][f].item(),2)
        train_metric_if = np.round(self.train_prediction_data[f"{output_name}_{self.metric_fn.name}"][f].item(),2) if self.metric_fn is not None else None
        if include_baseline:
          train_baseline_prediction_if = self.train_prediction_data[f"{output_name}_baseline_prediction"][:, f]
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

        ax_if.axvspan(train_time.min(), train_time.max(), facecolor='gray', alpha=0.2, label = train_label)

        if val_time is not None:
          val_target_if = self.val_prediction_data[f"{output_name}_actual"][:, f]
          val_prediction_if = self.val_prediction_data[f"{output_name}_prediction"][:, f]
          val_loss_if = np.round(self.val_prediction_data[f"{output_name}_{self.loss_fn.name}"][f].item(),2)
          val_metric_if = np.round(self.val_prediction_data[f"{output_name}_{self.metric_fn.name}"][f].item(),2) if self.metric_fn is not None else None
          if include_baseline:
            val_baseline_prediction_if = self.val_prediction_data[f"{output_name}_baseline_prediction"][:, f]
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

          ax_if.axvspan(val_time.min(), val_time.max(), facecolor='blue', alpha=0.2, label = val_label)

        if test_time is not None:
          test_target_if = self.test_prediction_data[f"{output_name}_actual"][:, f]
          test_prediction_if = self.test_prediction_data[f"{output_name}_prediction"][:, f]
          test_loss_if = np.round(self.test_prediction_data[f"{output_name}_{self.loss_fn.name}"][f].item(),2)
          test_metric_if = np.round(self.test_prediction_data[f"{output_name}_{self.metric_fn.name}"][f].item(),2) if self.metric_fn is not None else None
          if include_baseline:
            test_baseline_prediction_if = self.test_prediction_data[f"{output_name}_baseline_prediction"][:, f]
            test_baseline_loss_if = np.round(self.test_prediction_data[f"{output_name}_baseline_{self.loss_fn.name}"][f].item(),2)
            test_baseline_metric_if = np.round(self.test_prediction_data[f"{output_name}_baseline_{self.metric_fn.name}"][f].item(),2) if self.metric_fn is not None else None

          ax_if.plot(test_time, test_target_if, '-k')
          ax_if.plot(test_time, test_prediction_if, '-r')
          test_label = f"Test ({self.loss_fn.name} = {test_loss_if}, {self.metric_fn.name} = {test_metric_if})" \
                        if test_metric_if is not None \
                        else f"Test ({self.loss_fn.name} = {test_loss_if})"
          if include_baseline:
            ax_if.plot(test_time, test_baseline_prediction_if, '--g', linewidth = 1.0)
            test_label = test_label + f", Baseline ({self.loss_fn.name} = {test_baseline_loss_if}, {self.metric_fn.name} = {test_baseline_metric_if})"

          ax_if.axvspan(test_time.min(), test_time.max(), facecolor='red', alpha=0.2, label = test_label)

        if (f == 0) & (feature_names is not None):
          ax_if.set_title(output_name)
        if f == output_size[i] - 1:
          ax_if.set_xlabel(f"Time [{self.trainer.datamodule.time_unit}]")

        if feature_names is None:
          ylabel = f"{output_name} [{output_feature_units_if}]" if output_feature_units_if is not None else f"{output_name}"
        elif output_feature_name_if is not None:
          ylabel = f"{output_feature_name_if} [{output_feature_units_if}]" if output_feature_units_if is not None else f"{output_feature_name_if}"
        else:
          ylabel = f"[{output_feature_units_if}]" if output_feature_units_if is not None else None

        ax_if.set_ylabel(ylabel)

        ax_if.legend(loc='upper left', bbox_to_anchor=(1.02, 1), ncol=1) # loc = 'upper center', bbox_to_anchor = (0.5, 1.15), ncol = 5))
        ax_if.grid()

    if num_outputs > 1:
      for i in range(num_outputs, rows):
          ax[i].axis("off")

    fig.tight_layout()

    self.actual_prediction_plot = plt.gcf()
  ##

  ## forecast
  def forecast(self, num_forecast_steps = 1, hiddens = None):

    with torch.no_grad():
      steps = None

      if self.trainer.datamodule.test_dl is not None:
        for batch in self.trainer.datamodule.test_dl.dl: last_sample = batch
        data = self.trainer.datamodule.test_data
      elif self.trainer.datamodule.val_dl is not None:
        for batch in self.trainer.datamodule.val_dl.dl: last_sample = batch
        data = self.trainer.datamodule.val_data
      else:
        for batch in self.trainer.datamodule.train_dl.dl: last_sample = batch
        data = self.trainer.datamodule.train_data

      input, _, steps, batch_size = last_sample

      last_input_sample, last_steps_sample = input[:batch_size][-1:], steps[:batch_size][-1:]

      max_output_len = self.trainer.datamodule.max_output_len
      max_input_size, max_output_size = self.trainer.datamodule.max_input_size, self.trainer.datamodule.max_output_size
      output_mask = self.trainer.datamodule.train_output_mask
      output_input_idx, input_output_idx = self.trainer.datamodule.output_input_idx, self.trainer.datamodule.input_output_idx

      output, hiddens = self.forward(input = last_input_sample,
                                    steps = last_steps_sample,
                                    hiddens = hiddens,
                                    target = None,
                                    output_len = max_output_len,
                                    output_mask = output_mask)

      forecast = torch.empty((1, 0, max_output_size)).to(output)
      forecast_steps = torch.empty((1, 0)).to(last_steps_sample)

      input, steps = last_input_sample, last_steps_sample

      steps += max_output_len

      while forecast.shape[1] < num_forecast_steps:

        input_ = torch.zeros((1, max_output_len, max_input_size)).to(input)

        if len(output_input_idx) > 0:
          input_[:, :, output_input_idx] = output[:, -max_output_len:, input_output_idx]

        input = torch.cat((input[:, max_output_len:], input_), 1)

        output, hiddens = self.forward(input = input,
                                       steps = steps,
                                       hiddens = hiddens,
                                       target = None,
                                       output_len = max_output_len,
                                       output_mask = output_mask)

        forecast = torch.cat((forecast, output[:, -max_output_len:]), 1)
        forecast_steps = torch.cat((forecast_steps, steps[:, -max_output_len:]), 1)

        steps += max_output_len

      forecast, forecast_steps = forecast[:, -num_forecast_steps:], forecast_steps[:, -num_forecast_steps:]
      forecast_reduced, forecast_steps_reduced = self.generate_reduced_output(forecast, forecast_steps,
                                                                          transforms=self.trainer.datamodule.transforms)

      # self.forecast_data = {"warmup_time": }

    return forecast_reduced, forecast_steps_reduced


  ##
  def generate_reduced_output(self, output, output_steps, reduction='mean', transforms=None):

    # Get unique output steps and remove any -1 values
    unique_output_steps = output_steps.unique()
    unique_output_steps = unique_output_steps[unique_output_steps != -1]

    # Create a tensor to store the reduced output
    output_reduced = torch.zeros((len(unique_output_steps), np.sum(self.model.output_size))).to(output)

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
        for i in range(self.model.num_outputs):
            output_name_i = self.trainer.datamodule.output_names[i]
            output_reduced[:, j:(j + self.model.output_size[i])] = transforms[output_name_i].inverse_transform(output_reduced[:, j:(j + self.model.output_size[i])])
            j += self.model.output_size[i]

    # Return the reduced output and unique output steps
    return output_reduced, unique_output_steps

  def fit(self,
          datamodule,
          max_epochs = 20,
          callbacks = [None]):

    try:
      self.trainer = pl.Trainer(max_epochs = max_epochs,
                                accelerator = 'gpu' if self.model.device == 'cuda' else 'cpu',
                                callbacks = callbacks)

      self.trainer.fit(self,
                       datamodule = datamodule)

    except KeyboardInterrupt:
      state_dict = self.model.state_dict()
      self.model.to(device = self.model.device,
                    dtype = self.model.dtype)
      self.model.load_state_dict(state_dict)
