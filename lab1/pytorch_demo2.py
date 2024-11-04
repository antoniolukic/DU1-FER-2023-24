import torch
import numpy as np
from torch.utils.data import DataLoader

class Affine(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, input):
        return self.linear(input) + self.bias


affine = Affine(3, 4)
print(list(affine.named_parameters()))
print(affine(torch.randn(5, 3)))

#  dataset
dataset = [(torch.randn(4, 4), torch.randint(5, size=())) for _ in range(25)]
dataset = [(x.numpy(), y.numpy()) for x, y in dataset]
loader = DataLoader(dataset, batch_size=8, shuffle=False,
                    num_workers=0, collate_fn=None, drop_last=False)
for x, y in loader:
    print(x.shape, y.shape)


def supervised_training_step(ctx, x, y):
    ctx.model.train()  # postavljanje modela u stanje za učenje
    output = ctx.model(x)  # unaprijedni prolaz
    loss = ctx.loss(output, y).mean()  # izračun gubitka

    ctx.optimizer.zero_grad()  # postavljanje gradijenta u 0
    loss.backward()  # unatražni prolaz
    ctx.optimizer.step()  # primjena koraka optimizacije
