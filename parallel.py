import numpy
import multiprocessing
import timeit
import h5py
import random


def load_data(data_file):
    """ Read Data """
    D = h5py.File(data_file)
    X, Y_ = D['X'][()].transpose(), numpy.squeeze(D['Y'][()] - 1).astype(int)
    X = numpy.reshape(X, (X.shape[0], 28, 28, 1))
    Y = numpy.eye(10)[Y_]
    return {'X': X, 'Y': Y}

def pow2(x):
    return x**2

def pow3(x):
    return x**3

def pow4(x):
    return x**4

def augment(x):
    augment_fns = [pow2, pow3, pow4]
    for i in xrange(x.shape[0]):
        x[i] = random.choice(augment_fns)(x[i])
    return x

def get_batch(X, Y, batch_size):
    """ Samples a minibatch of size batch_size """
    num_examples = X.shape[0]
    batch_mask = numpy.random.choice(num_examples, batch_size)
    x_batch, y_batch = X[batch_mask], Y[batch_mask]
    x_batch = augment(x_batch)
    return x_batch, y_batch

def augment_worker(x):
    augment_fns = [pow2, pow3, pow4]
    return random.choice(augment_fns)(x)

def augment_parallel(x, workers):
    pool = multiprocessing.Pool(processes=workers)
    ilist = [i for i in x]
    results = pool.map(augment_worker, ilist)
    return results

def get_batch_parallel(X, Y, batch_size, workers=4):
    """ Samples a minibatch of size batch_size """
    num_examples = X.shape[0]
    batch_mask = numpy.random.choice(num_examples, batch_size)
    x_batch, y_batch = X[batch_mask], Y[batch_mask]
    x_batch = augment_parallel(x_batch, workers)
    return x_batch, y_batch

if __name__ == '__main__':
    print multiprocessing.cpu_count()
    dataset = load_data('MNIST.mat')
    get_batch(dataset['X'], dataset['Y'], batch_size=256)
    get_batch_parallel(dataset['X'], dataset['Y'], batch_size=256)
