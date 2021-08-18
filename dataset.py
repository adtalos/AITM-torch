from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class XDataset(Dataset):
  '''load csv data with feature name ad first row'''
  def __init__(self, datafile):
    super(XDataset, self).__init__()
    self.feature_names = []
    self.datafile = datafile
    self.data = []
    self._load_data()

  def _load_data(self):
    print("start load data from: {}".format(self.datafile))
    count = 0
    with open(self.datafile) as f:
      self.feature_names = f.readline().strip().split(',')[2:]
      for line in f:
        count += 1
        line = line.strip().split(',')
        line = [int(v) for v in line]
        self.data.append(line)
    print("load data from {} finished".format(self.datafile))

  def __len__(self, ):
    return len(self.data)

  def __getitem__(self, idx):
    line = self.data[idx]
    click = line[0]
    conversion = line[1]
    features = dict(zip(self.feature_names, line[2:]))
    return click, conversion, features
