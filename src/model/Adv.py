import torch
import torch.nn.functional as F
from model.roberta import roberta
from torch import nn
import numpy as np
from torch.autograd import Function

class MLP(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim=100):
    super(MLP, self).__init__()
    self.layers = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )

  def forward(self, x):
    x = self.layers(x)
    return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)

class ReverseLayerF(Function):

  @staticmethod
  def forward(ctx, x, alpha):
    ctx.alpha = alpha

    return x.view_as(x)

  @staticmethod
  def backward(ctx, grad_output):
    output = grad_output.neg() * ctx.alpha

    return output, None



class AdvModel(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.feature_extractor = roberta(args)
    # add new tokens
    self.prompt_k = args.prompt_k
    self.feature_extractor.tokenizer.add_tokens([f'<prompt_{x}>' for x in range(args.prompt_k * 2)], special_tokens=True)
    self.feature_extractor.model.resize_token_embeddings(len(self.feature_extractor.tokenizer))
    self.verbalizer = torch.tensor([self.feature_extractor.tokenizer('bad')['input_ids'][1], self.feature_extractor.tokenizer('good')['input_ids'][1]]).to('cuda')
    self.verbalizer = self.verbalizer.unsqueeze(-1).unsqueeze(-1)
    self.head = nn.Linear(768, args.num_classes)
    self.head.apply(weights_init)
    # self.domain_head = MLP(768, 2)
    self.domain_head = nn.Sequential(
                        nn.Linear(768, 768 // 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(768 // 2, 2),
                    )
    self.domain_head.apply(weights_init)
    self.loss_fn = nn.CrossEntropyLoss()
    self.alpha = args.alpha
    ''' optimization '''
    self.optimizer = torch.optim.AdamW(self._get_param_list([self.feature_extractor]), lr=args.lr)
    self.d_optimizer = torch.optim.AdamW(self._get_param_list([self.head, self.domain_head]), lr=args.domain_lr)

    self.mix_alpha_c = args.mix_alpha_c
    self.mix_alpha_d = args.mix_alpha_d
    self.epochs = args.epochs
  
  def my_CrossEntropyLoss(self, x, target, mix_lambda):
    x = F.log_softmax(x, dim = -1)
    nllloss = x.gather(1, target.view(-1, 1)) * mix_lambda
    nllloss = -nllloss.mean()
    return nllloss

  def _get_param_list(self, models):
    params_decay, params_no_decay = list(), list()
    for model in models:
      for name, param in model.named_parameters():
          if not param.requires_grad:
              continue
          if len(param.shape) == 1 or name.endswith('.bias'):
              params_no_decay.append(param)
          else:
              params_decay.append(param)
    param_list = [{'params': params_decay,
                    'weight_decay': self.args.weight_decay,
                    'initial_lr': self.args.lr},
                  {'params': params_no_decay,
                    'weight_decay': 0,
                    'initial_lr': self.args.lr}]
    return param_list
  
  def infer(self, x, labels):
    cls_feat, prompt_logits = self.feature_extractor(**x, ret_token=self.prompt_k + 1)
    label_words_logits = prompt_logits[:, self.verbalizer].squeeze(-1)
    pred = F.softmax(label_words_logits.reshape(label_words_logits.shape[0], -1), dim=-1).reshape(*label_words_logits.shape).squeeze(-1)
    acc = (torch.max(pred, -1)[1] == labels).sum() / labels.size(0) * 100
    loss = self.loss_fn(pred, labels)
    return pred, loss, acc

  def forward(self, x_s, x_s_unl, x_t, labels, epoch):
    if epoch > self.epochs // 2:
      cls_feat_s, cls_feat_t, prompt_logits_t, prompt_logits_mix, mix_lambda = self.feature_extractor(input_ids=x_s['input_ids'], attention_mask=x_s['attention_mask'], 
        input_ids_2=x_t['input_ids'], attention_mask_2=x_t['attention_mask'], ret_token=self.prompt_k + 1, mix_flag= True, verbalizer = self.verbalizer, labels=labels)

      prompt_logits_t = prompt_logits_t[:, self.verbalizer].squeeze(-1)
      prompt_logits_t = F.softmax(prompt_logits_t.reshape(prompt_logits_t.shape[0], -1), dim=-1).reshape(*prompt_logits_t.shape).squeeze(-1)
      psudo_label_t = torch.argmax(prompt_logits_t, dim=-1)
      
      prompt_logits_mix = prompt_logits_mix[:, self.verbalizer].squeeze(-1)
      prompt_logits_mix = F.softmax(prompt_logits_mix.reshape(prompt_logits_mix.shape[0], -1), dim=-1).reshape(*prompt_logits_mix.shape).squeeze(-1)
      prompt_logits_mix = torch.log(prompt_logits_mix+1e-15)
      acc = (torch.max(prompt_logits_mix, -1)[1] == labels).sum() / labels.size(0) * 100

      loss = self.my_CrossEntropyLoss(prompt_logits_mix, labels, mix_lambda) + self.my_CrossEntropyLoss(prompt_logits_mix, psudo_label_t, 1 - mix_lambda)

      r_s_feat = ReverseLayerF.apply(cls_feat_s, self.alpha)
      r_t_feat = ReverseLayerF.apply(cls_feat_t, self.alpha)
      domain_feat_st = self.domain_head(mix_lambda.squeeze(-1) * r_s_feat + (1 - mix_lambda.squeeze(-1)) * r_t_feat)
      domain_loss = self.my_CrossEntropyLoss(domain_feat_st, torch.zeros_like(labels), mix_lambda) + self.my_CrossEntropyLoss(domain_feat_st, torch.ones_like(labels), 1 - mix_lambda)

      loss = loss + domain_loss
    else:
      cls_feat_s, prompt_logits_s = self.feature_extractor(input_ids=x_s['input_ids'], attention_mask=x_s['attention_mask'], ret_token=self.prompt_k + 1)
      cls_feat_t, _ = self.feature_extractor(input_ids=x_t['input_ids'], attention_mask=x_t['attention_mask'], ret_token=self.prompt_k + 1) 

      label_words_logits = prompt_logits_s[:, self.verbalizer].squeeze(-1)
      label_words_logits = F.softmax(label_words_logits.reshape(label_words_logits.shape[0], -1), dim=-1).reshape(*label_words_logits.shape)
      label_words_logits = torch.log(label_words_logits+1e-15).squeeze(-1)
      acc = (torch.max(label_words_logits, -1)[1] == labels).sum() / labels.size(0) * 100
      loss = self.loss_fn(label_words_logits, labels)
            
      r_s_feat = ReverseLayerF.apply(cls_feat_s, self.alpha)
      r_t_feat = ReverseLayerF.apply(cls_feat_t, self.alpha)
      domain_feat_s = self.domain_head(r_s_feat)
      domain_feat_t = self.domain_head(r_t_feat)
      domain_loss = self.loss_fn(domain_feat_s, torch.zeros_like(labels)) + self.loss_fn(domain_feat_t, torch.ones_like(labels))
      loss = loss + domain_loss

    return loss, acc

  def step(self, loss):
    self.d_optimizer.zero_grad()
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.d_optimizer.step()

