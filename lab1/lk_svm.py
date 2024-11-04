from torchvision import datasets
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

dataset_root = r'C:\My_documents\8. semestar\Duboko ucenje 1\lab1'
mnist_train = datasets.MNIST(root=dataset_root, train=True, download=True)
mnist_test = datasets.MNIST(root=dataset_root, train=False, download=True)

x_train, y_train = mnist_train.data.float().div_(255.0), mnist_train.targets
x_test, y_test = mnist_test.data.float().div_(255.0), mnist_test.targets

x_train = x_train.view(x_train.shape[0], -1).numpy()
x_test = x_test.view(x_test.shape[0], -1).numpy()

# linear SVM
linear_svm = LinearSVC()
linear_svm.fit(x_train, y_train)
linear_svm_preds = linear_svm.predict(x_test)
linear_svm_accuracy = accuracy_score(y_test, linear_svm_preds)
print("Linear SVM Accuracy:", linear_svm_accuracy)  # 0.9185

# kernel SVM with RBF kernel
kernel_svm = SVC(kernel='rbf')
kernel_svm.fit(x_train, y_train)
kernel_svm_preds = kernel_svm.predict(x_test)
kernel_svm_accuracy = accuracy_score(y_test, kernel_svm_preds)
print("Kernel SVM Accuracy:", kernel_svm_accuracy)  # 0.9792
