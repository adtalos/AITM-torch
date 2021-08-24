from torch.utils.data import DataLoader
from dataset import XDataset

datafile = './data/ctr_cvr.dev'
dataset = XDataset(datafile)

dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
for i, value in enumerate(dataloader):
  click, conversion, features = value
  print(click.shape)
  print(conversion.shape)
  for key in features.keys():
    print(key, features[key].shape)
