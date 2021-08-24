import torch
from model import AITM

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

embedding_size = 5
model_file = './out/AITM.model'
model = AITM(vocabulary_size, embedding_size)
model.load_state_dict(torch.load(model_file))
example = {key: torch.tensor([[0]]) for key in vocabulary_size}
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("./out/AITM.model.pt")
