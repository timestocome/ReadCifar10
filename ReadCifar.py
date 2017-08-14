
# http://github.com/timestocome


# data 
# https://www.cs.toronto.edu/~kriz/cifar.html



import numpy as np
import pickle
import matplotlib.pyplot as plt

###################################################################################
# read in data
##################################################################################
n_classes = 10
image_height = 32
image_width = 32
image_depth = 3
label_bytes = 1


def unpickle(file):

    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict



def load_data():

    xs = []
    ys = []
    
    # read in training files
    for i in range(5):
        # this is the directory you put the cifar batch files into
        filename = 'cifar-10/data_batch_%d' % (i+1)
        with open(filename, 'rb') as f:
    
            d = pickle.load(f, encoding='latin1') # needed for python2-python3 pickle
            x = d['data']
            y = d['labels']
            xs.append(x)
            ys.append(y)

    # read in test files
    filename = 'cifar-10/test_batch'
    with open(filename, 'rb') as f:

        d = pickle.load(f, encoding='latin1')
        xs.append(d['data'])
        ys.append(d['labels'])

    
    x = np.concatenate(xs)                      # images
    y = np.concatenate(ys)                      # labels
    x = x.reshape((x.shape[0], 3, 32, 32)).transpose(0,2,3,1)

    
    # Visualizing CIFAR 10
    fig, axes1 = plt.subplots(5,5,figsize=(10,10))
    for j in range(5):
        for k in range(5):
            i = np.random.choice(range(len(x)))
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(x[i:i+1][0])

    plt.show()

    
    # scale images
    x = x / 255.



load_data()