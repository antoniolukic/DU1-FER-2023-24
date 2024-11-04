from __future__ import annotations
from pt_deep import *
import torch
import torchvision
import os
from sklearn.model_selection import train_test_split


def get_data():
    dataset_root = r'C:\My_documents\8. semestar\Duboko ucenje 1\lab1'  # change this to your preference
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=False)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=False)
    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets
    x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)
    return x_train, x_test, y_train, y_test


def preprocessing(x_train, x_test, y_train, y_test):
    x_train = x_train.double().view(len(x_train), -1)  # flatten
    x_test = x_test.double().view(len(x_test), -1)

    x_train = x_train.to(device)  # move to device
    x_test = x_test.to(device)

    y_train_oh = torch.tensor(class_to_onehot(y_train)).to(device)  # one-hot
    y_test_oh = torch.tensor(class_to_onehot(y_test)).to(device)
    return x_train, x_test, y_train, y_test, y_train_oh, y_test_oh


def save_model_and_loss(model, loss_history, save_dir, model_filename='model.pth', loss_filename='loss_history.npy'):
    model_path = os.path.join(save_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    loss_history = [item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item for item in loss_history]
    loss_path = os.path.join(save_dir, loss_filename)
    np.save(loss_path, loss_history)


def plot_loss_history(loss_filename_1, loss_filename_2, label_1='Loss 1', label_2='Loss 2'):
    loss_history_1 = np.load(loss_filename_1)
    loss_history_2 = np.load(loss_filename_2)
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history_1, label=label_1)
    plt.plot(loss_history_2, label=label_2)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss History Comparison')
    plt.legend()
    plt.show()


def plot_weight_matrices(model):
    weights = model.weights[0].detach().cpu().numpy().T  # get the weights and transpose for plotting
    num_digits = weights.shape[0]
    fig, axs = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(num_digits):
        ax = axs[i // 5, i % 5]
        ax.imshow(weights[i].reshape(28, 28), cmap='gray')
        ax.set_title(f'Weight Matrix for Digit {i}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def train_with_validation(model: PTDeep, x_train, y_train_oh, x_val, y_val_oh, max_epochs, learning_rate, regularization):
    best_model_state = None
    best_val_loss = float('inf')

    train_loss_history = []
    val_loss_history = []

    for epoch in range(max_epochs):
        epoch_loss = train(model, x_train, y_train_oh, 1, learning_rate, regularization)  # train
        train_loss_history.extend(epoch_loss)

        with torch.no_grad():  # evaluate
            model.eval()
            val_loss = model.get_loss(x_val, y_val_oh, regularization)
            model.train()

        val_loss_history.append(val_loss)

        if val_loss < best_val_loss:  # update if valid lower
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

    if best_model_state:  # return best model
        best_model = PTDeep(model.layers, model.activation_fun)
        best_model.load_state_dict(best_model_state)
        return best_model, train_loss_history, val_loss_history
    else:
        return model, train_loss_history, val_loss_history


if __name__ == "__main__":
    # 5000 0.2 0.01 [784, 10], 5000 0.05 0.01 [784, 100, 10]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(100)

    x_train, x_test, y_train, y_test = get_data()
    x_train, x_test, y_train, y_test, y_train_oh, y_test_oh = preprocessing(x_train, x_test, y_train, y_test)

    ptdeep = PTDeep([784, 100, 10], "relu")  # define model
    ptdeep.to(device)
    ptdeep.count_params()
    #  print("Loss of a random initialized model(baseline): ", ptdeep.get_loss(x_test, y_test_oh, 0.01).item())
    loss_history = train(ptdeep, x_train, y_train_oh, 5000, 0.05, 0.01)  # train

    #  x_train, x_valid, y_train_oh, y_valid_oh = train_test_split(x_train, y_train_oh, test_size=1/5, random_state=42)
    #  model, train_loss_history, val_loss_history = train_with_validation(ptdeep, x_train, y_train_oh, x_valid, y_valid_oh, 5000, 0.05, 0.01)
    #  train_loss_history = [tensor.item() for tensor in train_loss_history]
    #  val_loss_history = [tensor.item() for tensor in val_loss_history]


    #  loss_history = train_mb(ptdeep, x_train, y_train_oh, 100, 0.001, 0.01)
    #  loss_history = train_adam(ptdeep, x_train, y_train_oh, 1000, 0.01, 0.01)  # ne postize nikakve bas rezultate

    #  loss_history = train_adam_auto(ptdeep, x_train, y_train_oh, 1000, 0.01, 0.01)  # train
    #  loss_np_list = [tensor.cpu().detach().numpy() if tensor.dim() > 0 else np.array(tensor.cpu().detach()) for tensor in loss_history]
    #  loss_np = np.stack(loss_np_list)

    Y = eval(ptdeep, x_test.cpu().numpy())  # probabilities for train set
    accuracy, pr, M = eval_perf_multi(Y, y_test.cpu())
    print(accuracy); print(pr); print(M)

    #  plot_weight_matrices(ptdeep)
    #  save_model_and_loss(ptdeep, loss_history, save_dir='saved_model', model_filename='model_2.pth', loss_filename='loss_history_2.npy')
    #  plot_loss_history('saved_model/loss_history_1.npy', 'saved_model/loss_history_2.npy')
    #  ptdeep.plot_highest_loss_indexes(x_test, y_test_oh)
