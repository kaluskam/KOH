import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from mnist import MNIST

from network_higher_dims import Network


mndata = MNIST('data\\mnist-test-images')
mndata.test_img_fname = 't10k-images-idx3-ubyte\\t10k-images.idx3-ubyte'
mndata.test_lbl_fname = 't10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte'
images, labels = mndata.load_testing()

images = np.array(images)
labels = np.array(labels)

# KN = Network(input_shape=len(images[0]), shape=(2, 5), grid_type='rect', name='mnist_500_rect')

# #KN.name = 'KN_mnist_100_rect'
# KN.fit(x_train=images, y_train=labels, start_epoch=0, n_epochs=100, neighbourhood_scale=1)
# print(KN.evaluate())
# KN.plot()
#
KN = Network(input_shape=len(images[0]), shape=(2, 5), grid_type='rect', name='mnist_100_rect_0_6')
KN.fit(x_train=images, y_train=labels, n_epochs=100, neighbourhood_scale=0.6)
print(KN.evaluate())
KN.plot()

KN = Network(input_shape=len(images[0]), shape=(2, 5), grid_type='hex', name='mnist_100_hex')
KN.fit(x_train=images, y_train=labels, n_epochs=100, neighbourhood_scale=1)
print(KN.evaluate())
KN.plot()

KN = Network(input_shape=len(images[0]), shape=(2, 5), grid_type='hex', name='mnist_100_hex_0_6')
KN.fit(x_train=images, y_train=labels, n_epochs=100, neighbourhood_scale=0.6)
print(KN.evaluate())
KN.plot()
