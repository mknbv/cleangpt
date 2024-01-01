"""Training GPT models."""
from tqdm.auto import tqdm, trange
from torch.optim import AdamW
import torch.nn.utils
from cleangpt.layers import CrossEntropy


class Trainer:
  """GPT module trainer."""
  def __init__(self, model, optimizer=None, weight_decay=0.1, grad_clip=1.0,
               **optimizer_kwargs):
    self.model = model
    self.xent = CrossEntropy()
    self.grad_clip = grad_clip
    if optimizer is None:
      optimizer_kwargs = dict(lr=5e-4, betas=(0.9, 0.95)) | optimizer_kwargs
      optimizer = AdamW(model.param_groups(weight_decay), **optimizer_kwargs)
    self.optimizer = optimizer

  @property
  def device(self):
    """Returns the device used for training."""
    return next(self.model.parameters()).device

  def loss(self, logits, targets):
    """Computes the loss function value."""
    return self.xent(logits.transpose(1, 2), targets)

  def iter_epoch(self, data_loader, leave_tqdm=False):
    """Training epoch iterator."""
    self.model.train()
    for inputs, targets in tqdm(data_loader, leave=leave_tqdm):
      inputs, targets = inputs.to(self.device), targets.to(self.device)
      outputs = self.model(inputs)
      loss = self.loss(outputs, targets)

      self.optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
      yield inputs, targets, loss
      self.optimizer.step()

  def train(self, data_loader, nepochs=100):
    """Performs training for the specified number of epochs."""
    for _ in trange(nepochs):
      for _ in self.iter_epoch(data_loader):
        pass
