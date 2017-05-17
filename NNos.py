import lasagne
import theano
import theano.tensor as T
import numpy as np
import time

input_var = T.tensor4('X')
target_var = T.tensor4('y')
size =  100,100
num_epochs=100


def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, size[0], size[1]),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=128, filter_size=(3, 3), pad='same',
        nonlinearity=lasagne.nonlinearities.rectify)

    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=512, filter_size=(3, 3), pad='same',
        nonlinearity=lasagne.nonlinearities.rectify)

    #network = lasagne.layers.Conv2DLayer(
    #    network, num_filters=32, filter_size=(5, 5),
    #    nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.Upscale2DLayer(
        network,scale_factor=2)

    network = lasagne.layers.TransposedConv2DLayer(
        network, num_filters=128, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.Upscale2DLayer(
        network,scale_factor=2)

    network = lasagne.layers.TransposedConv2DLayer(
        network, num_filters=64, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.Upscale2DLayer(
        network,scale_factor=2)

    network = lasagne.layers.TransposedConv2DLayer(
        network, num_filters=16, filter_size=(5, 5),crop='same',
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=1, filter_size=(1, 1),
        nonlinearity=lasagne.nonlinearities.sigmoid)
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    #network = lasagne.layers.DenseLayer(
    #        lasagne.layers.dropout(network, p=.5),
    #        num_units=256,
    #        nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    #network = lasagne.layers.DenseLayer(
    #        lasagne.layers.dropout(network, p=.5),
    #        num_units=10,
    #        nonlinearity=lasagne.nonlinearities.softmax)

    return network


def build_cnn_small(input_var=None,size=(100,100)):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, size[1], size[0]),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.


    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=16, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=16, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)


    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    #network = lasagne.layers.Conv2DLayer(
    #    network, num_filters=512, filter_size=(3, 3), pad='same',
    #    nonlinearity=lasagne.nonlinearities.rectify)

    #network = lasagne.layers.Conv2DLayer(
    #    network, num_filters=32, filter_size=(5, 5),
    #    nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.Upscale2DLayer(
        network,scale_factor=2)

    network = lasagne.layers.TransposedConv2DLayer(
        network, num_filters=32, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.Upscale2DLayer(
        network,scale_factor=2)

    network = lasagne.layers.TransposedConv2DLayer(
        network, num_filters=16, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify)

    #network = lasagne.layers.Upscale2DLayer(
    #    network,scale_factor=2)

    network = lasagne.layers.TransposedConv2DLayer(
        network, num_filters=16, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=1, filter_size=(1, 1),
        nonlinearity=lasagne.nonlinearities.sigmoid)
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    #network = lasagne.layers.DenseLayer(
    #        lasagne.layers.dropout(network, p=.5),
    #        num_units=256,
    #        nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    #network = lasagne.layers.DenseLayer(
    #        lasagne.layers.dropout(network, p=.5),
    #        num_units=10,
    #        nonlinearity=lasagne.nonlinearities.softmax)

    return network

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    print '=='
    for inp in inputs:
        print 'inp ',inp.shape
    #shuffle=True#added
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
            print 'exc ', excerpt
            yield [inputs[i] for i in excerpt], [targets[i] for i in excerpt]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
            print 'exc2 ', excerpt
            yield inputs[excerpt], targets[excerpt]

 #       yield [inputs[i] for i in excerpt], [targets[i] for i in excerpt]#inputs[excerpt], targets[excerpt]


def train(X_train,X_val,X_test,y_train,y_val,y_test,size):
# Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    network=build_cnn_small(input_var=input_var,size=size)
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            print len(inputs),len(targets)
            for inp in inputs:
                print 'in ',inp.shape
            for tar in targets:
                print 'tr ',tar.shape
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 1, shuffle=False):
            inputs, targets = batch
            #print 'inpu ',inputs[0]
            #print 'targu ',targets[0]
            for x in inputs:
                print 'X_vi ',x.shape
            for y in targets:
                print 'yvt ',y.shape

            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=True):#false
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))