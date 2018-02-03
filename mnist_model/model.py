from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.callbacks import TensorBoard

from time import strftime
from os import path

class MNIST_model:
    """ Wrapper for a keras model for mnist. """

    def __init__(self, img_size=(28,28,1),
                input_shape=(784,),
                arch='c-32-5_p-2_c-64-5_p-2_f_d-1024_o-4',
                conv_activation='relu',
                optimizer='rmsprop',
                loss='categorical_crossentropy',
                batch_size=128,
                epochs=20,
                debug_graph=False):
        """ Create a model for mnist with the given architecture """
        model = Sequential()
        model.add(Reshape(img_size, input_shape=input_shape))
        for l in arch.split('_'):
            if l.startswith('c'):
                filters = int(l.split('-')[1])
                kernel_size = int(l.split('-')[2])
                model.add(Conv2D(filters, (kernel_size,kernel_size), activation=conv_activation))
            elif l.startswith('p'):
                size = int(l.split('-')[1])
                model.add(MaxPooling2D((size,size)))
            elif l.startswith('f'):
                model.add(Flatten())
            elif l.startswith('d'):
                size = int(l.split('-')[1])
                model.add(Dense(size))
            elif l.startswith('o'):
                factor = int(l.split('-')[1]) / 10
                model.add(Dropout(factor))
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        self.model = model
        self.name = '{}__{}__{}_{}__{}_{}'.format(arch, conv_activation, optimizer, loss, batch_size, epochs)
        self.batch_size = batch_size
        self.epochs = epochs
        self.debug_graph = debug_graph

    def train(self,x,y,val_data,log_dir):
        self.timestamp = strftime('%Y-%m-%d_%H%M%S')
        model_log = path.join(log_dir, path.join(self.name, self.timestamp))
        tensorboard = TensorBoard(log_dir=model_log,
                                    write_graph=self.debug_graph,
                                    write_images=True)
        self.model.fit(x=x,y=y,validation_data=val_data,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[tensorboard])

    def summary(self):
        self.model.summary()

    def save(self, directory):
        self.model.save(path.join(directory, self.name + '__' + self.timestamp + '.hd5'))
