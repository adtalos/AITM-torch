import sys
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
import dataset
from model import AITM
import numpy as np
from sklearn.metrics import roc_auc_score


# super parameter
batch_size = 2000
embedding_size = 5
learning_rate = 0.0001
total_epoch = 10
earlystop_epoch = 1

vocabulary_size = {
    '101': 238635,
    '121': 98,
    '122': 14,
    '124': 3,
    '125': 8,
    '126': 4,
    '127': 4,
    '128': 3,
    '129': 5,
    '205': 467298,
    '206': 6929,
    '207': 263942,
    '216': 106399,
    '508': 5888,
    '509': 104830,
    '702': 51878,
    '853': 37148,
    '301': 4
}

model_file = './out/AITM.model'


def get_dataloader(filename, batch_size, shuffle):
  data = dataset.XDataset(filename)
  loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
  return loader


def train():
  train_dataloader = get_dataloader('./data/ctr_cvr.train',
                                    batch_size,
                                    shuffle=True)
  dev_dataloader = get_dataloader('./data/ctr_cvr.dev',
                                  batch_size,
                                  shuffle=True)
  model = AITM(vocabulary_size, embedding_size)
  device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=learning_rate,
                               weight_decay=1e-6)
  model.to(device)
  best_acc = 0.0
  earystop_count = 0
  best_epoch = 0
  for epoch in range(total_epoch):
    total_loss = 0.
    nb_sample = 0
    # train
    model.train()
    for step, batch in enumerate(train_dataloader):
      click, conversion, features = batch
      for key in features.keys():
        features[key] = features[key].to(device)
      click_pred, conversion_pred = model(features)
      loss = model.loss(click.float(),
                        click_pred,
                        conversion.float(),
                        conversion_pred,
                        device=device)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss += loss
      nb_sample += click.shape[0]
      if step % 200 == 0:
        print('[%d] Train loss on step %d: %.6f' %
              (nb_sample, (step + 1), total_loss / (step + 1)))

    # validation
    print("start validation...")
    click_pred = []
    click_label = []
    conversion_pred = []
    conversion_label = []
    model.eval()
    for step, batch in enumerate(dev_dataloader):
      click, conversion, features = batch
      for key in features.keys():
        features[key] = features[key].to(device)

      with torch.no_grad():
        click_prob, conversion_prob = model(features)

      click_pred.append(click_prob.cpu())
      conversion_pred.append(conversion_prob.cpu())

      click_label.append(click)
      conversion_label.append(conversion)

    click_auc = cal_auc(click_label, click_pred)
    conversion_auc = cal_auc(conversion_label, conversion_pred)
    print("Epoch: {} click_auc: {} conversion_auc: {}".format(
        epoch + 1, click_auc, conversion_auc))

    acc = click_auc + conversion_auc
    if best_acc < acc:
      best_acc = acc
      best_epoch = epoch + 1
      torch.save(model.state_dict(), model_file)
      earystop_count = 0
    else:
      print("train stop at Epoch %d based on the base validation Epoch %d" %
            (epoch + 1, best_epoch))
      return


def test():
  print("Start Test ...")
  test_loader = get_dataloader('./data/ctr_cvr.test',
                               batch_size=batch_size,
                               shuffle=False)
  model = AITM(vocabulary_size, 5)
  model.load_state_dict(torch.load(model_file))
  model.eval()
  click_list = []
  conversion_list = []
  click_pred_list = []
  conversion_pred_list = []
  for i, batch in enumerate(test_loader):
    if i % 1000:
      sys.stdout.write("test step:{}\r".format(i))
      sys.stdout.flush()
    click, conversion, features = batch
    with torch.no_grad():
      click_pred, conversion_pred = model(features)
    click_list.append(click)
    conversion_list.append(conversion)
    click_pred_list.append(click_pred)
    conversion_pred_list.append(conversion_pred)
  click_auc = cal_auc(click_list, click_pred_list)
  conversion_auc = cal_auc(conversion_list, conversion_pred_list)
  print("Test Resutt: click AUC: {} conversion AUC:{}".format(
      click_auc, conversion_auc))


def cal_auc(label: list, pred: list):
  label = torch.cat(label)
  pred = torch.cat(pred)
  label = label.detach().numpy()
  pred = pred.detach().numpy()
  auc = roc_auc_score(label, pred, labels=np.array([0.0, 1.0]))
  return auc


if __name__ == "__main__":
  train()
  test()
