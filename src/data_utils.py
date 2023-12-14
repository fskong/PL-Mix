import os
from functools import partial

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split


class TextDataset(Dataset):
  def __init__(self, text, label):
    self.text = text
    self.label = label

  def __getitem__(self, index):
    return self.text[index], self.label[index]
  
  def __len__(self):
    return len(self.text)

def readReview(path):
  val = {
    '1.0': 0,
    '2.0': 0,
    '3.0': 1,
    '4.0': 2,
    '5.0': 2
  }
  with open(path + '_text.txt') as f:
    text = f.readlines()
  with open(path + '_label.txt') as f:
    label = f.readlines()
  text = [t.strip() for t in text]
  label = list(map(lambda x: val[x.strip()], label))
  assert len(text) == len(label)
  return TextDataset(text, label)

def readAmazon(path):
  pos_path = path + 'pos.txt'
  neg_path = path + 'neg.txt'
  ret_text = []
  ret_label = []
  with open(pos_path) as f:
    text = f.readlines()
    text = [t.strip() for t in text]
    label = [1] * len(text)
    ret_text += text
    ret_label += label
  with open(neg_path) as f:
    text = f.readlines()
    text = [t.strip() for t in text]
    label = [0] * len(text)
    ret_text += text
    ret_label += label
  return TextDataset(ret_text, ret_label)


def collate_batch(batch, tokenizer, add_k=0, flag=None):
  text, label = [], []
  for t, l in batch:
    if add_k:
      if flag == 'source':
        prompt_sent = ''.join([f'<prompt_{x}>' for x in range(add_k)]) +  ' <mask> . '
      else:
        prompt_sent = ''.join([f'<prompt_{x}>' for x in range(add_k, add_k * 2)]) +  ' <mask> . '
      t = prompt_sent + t
    text.append(t)
    label.append(l)
  # print (text)
  text = tokenizer(text, max_length=512, return_tensors="pt", padding=True, truncation=True)
  input_ids_pad = torch.ones([text['input_ids'].size(0), 512 - text['input_ids'].size(1)], dtype = text['input_ids'].dtype)
  attention_mask_pad = torch.zeros([text['attention_mask'].size(0), 512 - text['attention_mask'].size(1)], dtype = text['attention_mask'].dtype)
  text['input_ids'] = torch.concat([text['input_ids'], input_ids_pad], -1)
  text['attention_mask'] = torch.concat([text['attention_mask'], attention_mask_pad], -1)
  label = torch.tensor(label)
  return text, label

def build_dataset(args, tokenizer):
  folder_name = 'amazon'
  if 'Review' in args.dataset:
    folder_name = args.dataset.split('-')[-1]
  data_dir = os.path.join(args.data_dir, folder_name)
  if 'Review' in args.dataset:
    source_data = readReview(os.path.join(data_dir, args.source, 'set1'))
    source_data, val_data = random_split(source_data, [len(source_data) - 1000, 1000])
    source_test_data = readReview(os.path.join(data_dir, args.source, 'set2'))
    unlabel_data = readReview(os.path.join(data_dir, args.target, 'set2'))
    test_data = readReview(os.path.join(data_dir, args.target, 'set1'))
  else:
    source_data = readAmazon(os.path.join(data_dir, args.source, ''))
    source_data, val_data = random_split(source_data, [len(source_data) - 400, 400])
    source_test_data = source_data
    unlabel_data = readAmazon(os.path.join(data_dir, args.target, 'un_'))
    # if len(unlabel_data) > 20000:
    unlabel_data, _ = random_split(unlabel_data, [4000, len(unlabel_data)-4000])
    test_data = readAmazon(os.path.join(data_dir, args.target, ''))
  add_k = 0
  if args.method == 'ADV':
    add_k = args.prompt_k
  # source split train
  train_loader = DataLoader(source_data, batch_size=args.train_batch_size, shuffle=True, drop_last=True, collate_fn=partial(collate_batch, tokenizer=tokenizer, add_k=add_k, flag='source'))
  # target unlabel split
  target_loader = DataLoader(unlabel_data, batch_size=args.train_batch_size, shuffle=True, drop_last=True, collate_fn=partial(collate_batch, tokenizer=tokenizer, add_k=add_k, flag='target'))
  # source split val
  val_loader = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False, collate_fn=partial(collate_batch, tokenizer=tokenizer, add_k=add_k, flag='source'))
  # target label
  test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, collate_fn=partial(collate_batch, tokenizer=tokenizer, add_k=add_k, flag='target'))
  # source split train
  source_test_loader = DataLoader(source_test_data, batch_size=args.train_batch_size, shuffle=True, drop_last=True, collate_fn=partial(collate_batch, tokenizer=tokenizer, add_k=add_k, flag='source'))
  return train_loader, target_loader, val_loader, source_test_loader, test_loader
  
