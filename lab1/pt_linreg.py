from __future__ import annotations
import torch
import torch.optim as optim


## Definicija računskog grafa
# podaci i parametri, inicijalizacija parametara
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

X = torch.tensor([1, 2, 3, 4, 5])
Y = torch.tensor([3, 5, 7, 9, 11])
N = len(X)
epochs = 100
eta = 0.01

# optimizacijski postupak: gradijentni spust
optimizer = optim.SGD([a, b], lr=eta)

for i in range(epochs):
    # afin regresijski model
    Y_ = a * X + b

    diff = (Y - Y_)

    # kvadratni gubitak
    loss = torch.sum(diff ** 2) / N

    # računanje gradijenata
    loss.backward()

    hand_a = 1 / N * torch.sum(-2 * (Y - Y_) * X)
    hand_b = 1 / N * torch.sum(-2 * (Y - Y_))

    # korak optimizacije
    optimizer.step()

    if i % 10 == 0:
        print(f'step: {i}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')
        print(f'a_grad: {a.grad}, b_grad: {b.grad}')
        print(f'a_hand: {hand_a}, b_hand: {hand_b}')
        print()

    # Postavljanje gradijenata na nulu
    optimizer.zero_grad()
