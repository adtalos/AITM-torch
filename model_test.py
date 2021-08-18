import model
import torch


def test_model():

  feature_vocab = {"0": 10, "1": 12, "2": 20}
  embedding_size = 4
  m = model.AITM(feature_vocab, embedding_size)
  inputs = {
      "0": torch.tensor([[1], [2]]),
      "1": torch.tensor([[2], [3]]),
      "2": torch.tensor([[10], [11]])
  }
  click, conversion = m(inputs)
  print("click_pred:", click.shape)
  print("covnersion_pred:", conversion.shape)

  click_label = torch.tensor([1.0, 1.0])
  conversion_label = torch.tensor([1.0, 0.0])

  loss = m.loss(click_label, click, conversion_label, conversion)
  print("loss: ", loss)


if __name__ == "__main__":
  test_model()
