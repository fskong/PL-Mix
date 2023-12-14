import time
from copy import deepcopy

import torch
from config.config import getarg
from data_utils import build_dataset
from utils import AverageMeter, CircleDataIter, ProgressMeter

from datetime import datetime 

class Instructor:
  def __init__(self, args):
    self.epoch = 0
    ''' build model '''
    num_classes = {"Review-small": 3, "Review-large":3, "Benchmark": 2}
    args.num_classes = num_classes[args.dataset]
    self.args = args
    self.model = args.model_class(args)
    self.model.to(args.device)

    ''' dataset '''
    self.train_source_loader, self.train_target_loader, self.val_loader, self.test_source_loader, self.test_loader = build_dataset(args, self.model.feature_extractor.tokenizer)
    self._print_args()
    # self.logger.watch(self.trainer.model)
  
  def _print_args(self):
    print('Training arguments:')
    for arg in vars(self.args):
      print(f">>> {arg}: {getattr(self.args, arg)}")

  def save_state_dict(self):
    torch.save({'model': self.model.state_dict() if hasattr(self, 'model') else None,
                'epoch': self.epoch,
                'args': self.args}, os.path.join(self.args.output_dir, 'checkpoint.pth'))
  
  def _validate(self, val_loader, setName='Test'):
    batch_time = AverageMeter('Time', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':5.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, cls_accs],
        prefix=f"{setName}: ")
    self.model.eval()

    with torch.no_grad():
      end = time.time()
      for i_batch, inputs in enumerate(val_loader):
        inputs = [x.to(self.args.device) for x in inputs]
        x, labels = inputs
        pred, loss, acc = self.model.infer(x, labels=labels)

        losses.update(loss.item(), labels.size(0))
        cls_accs.update(acc.item(), labels.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i_batch % self.args.print_freq == 0:
          progress.display(i_batch)
      print(f' * Acc = {cls_accs.avg: .3f} Loss = {losses.avg: .3f}')
    return cls_accs.avg

  def _train(self, train_source_iter, train_source_unl_iter, train_target_iter):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':5.2f')
    progress = ProgressMeter(
        self.args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(self.epoch))
    # switch to train mode
    self.model.train()
    end = time.time()
    for i in range(self.args.iters_per_epoch):
      data_time.update(time.time() - end)
      x_s, labels_s = next(train_source_iter)
      x_t, _ = next(train_target_iter)
      x_s_unl, _ = next(train_source_unl_iter)
      x_s = x_s.to(self.args.device)
      x_t = x_t.to(self.args.device)
      x_s_unl = x_s_unl.to(self.args.device)
      labels_s = labels_s.to(self.args.device)

      loss, acc = self.model(x_s, x_s_unl, x_t, labels=labels_s, epoch=self.epoch)

      losses.update(loss.item(), labels_s.size(0))
      cls_accs.update(acc.item(), labels_s.size(0))

      self.model.step(loss)

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % self.args.print_freq == 0:
        progress.display(i)

  def run(self):
    train_source_iter = CircleDataIter(self.train_source_loader)
    train_source_unl_iter = CircleDataIter(self.test_source_loader)
    train_target_iter = CircleDataIter(self.train_target_loader)
    best_acc = 0.
    for _ in range(self.args.epochs):
      self.epoch += 1
      # source split train,  source split train,  target unlabel split
      self._train(train_source_iter, train_source_unl_iter, train_target_iter)
      # source split val
      val_acc = self._validate(self.val_loader, setName="Validation")
      # target label
      test_acc = self._validate(self.test_loader, setName="Test")
      if val_acc > best_acc:
        best_model = deepcopy(self.model.state_dict())
        best_acc = val_acc
    
    print(f"best_acc = {best_acc: 5.2f}")
    self.model.load_state_dict(best_model)
    acc = self._validate(self.test_loader)
    print(f"test_acc = {acc: 5.2f}")
    return acc


def main():
  args = getarg()
  ins = Instructor(args)
  acc = ins.run()
  return acc


if __name__ == '__main__':
  start = datetime.now()
  main()
  end = datetime.now()
  print ((end - start).seconds)
