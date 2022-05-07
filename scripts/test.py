# read hparams from event file
from tbparse import SummaryReader
import torch

model = torch.nn.Sequential(
    torch.nn.Linear(2, 1),
    torch.nn.Sigmoid()
)

torch.save(model, "model.pt")

blah = torch.load("model.pt")

print(blah)
print(model)
print(model == blah)
breakpoint()
