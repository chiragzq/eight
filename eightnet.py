import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import struct
import threading

class TrainingProgressHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.epochs = 0

    def on_epoch_end(self, logs, logs2):
        self.epochs += 1

    def set_total_epochs(self, total):
        self.total_epochs = total
"""
class TrainThread(threading.Thread):
    def __init__(self, iterations, model, images, labels, callback):
        threading.Thread.__init__(self)
        self.iterations = iterations
        #self.model = model
        #self.images = images
        #self.labels = labels
        self.callback = callback

    def run(self):
        EightNet.model.fit(EightNet.images, EightNet.labels, epochs=self.iterations, callbacks=[self.callback])
"""
graph = tf.get_default_graph()
class EightNet:
    labels = None
    images = None
    initialLearningRate = 0.06
    model = None
    history = TrainingProgressHistory()
    thread = None
    
    def load_data(num_images, labelPath="train-labels-idx1-ubyte", imagePath="train-images-idx3-ubyte"):
        tempLabels = []
        tempImages = []
        with open(labelPath, "rb") as f:
            magic = struct.unpack('>i', f.read(4))[0]
            num_items = struct.unpack(">i", f.read(4))[0]
            for i in range(0, min(num_images, num_items)):
                tempLabels.append(struct.unpack("B", f.read(1))[0])
        print("Loaded %d/%d labels" % (min(num_images, num_items), num_items))
        
        with open(imagePath, "rb") as f:
            magic = struct.unpack('>i', f.read(4))[0]
            num_items = struct.unpack('>i', f.read(4))[0]
            num_rows = struct.unpack('>i', f.read(4))[0]
            num_columns = struct.unpack('>i', f.read(4))[0]
            for i in range(0, min(num_images, num_items)):
                tempImages.append([])
                for j in range(0, num_rows):
                    tempImages[i].append([])
                    for k in range(0, num_columns):
                        tempImages[i][j].append(struct.unpack("B", f.read(1))[0])
                if i % 25 == 24:
                    print("read %d images\r" % (i + 1))
        print("Loaded %d/%d images" % (min(num_images, num_items), num_items))
        for i in range(0, len(tempLabels)):
            arr = np.zeros(10)
            arr[tempLabels[i]] = 1
            tempLabels[i] = arr
        EightNet.labels = np.array(tempLabels)
        EightNet.images = np.array(tempImages) / 255.0
        
    def initialize_model():
        with graph.as_default():
            EightNet.model = keras.Sequential([
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Dense(128, activation=tf.nn.relu),
                keras.layers.Dense(10, activation=tf.nn.relu)
            ])
            EightNet.model.compile(optimizer=tf.keras.optimizers.SGD(EightNet.initialLearningRate), loss="mean_squared_error", metrics=["accuracy"])
    
    def train_model(iterations):
        with graph.as_default():
            if EightNet.thread == None or not EightNet.thread.isAlive():
                EightNet.history.set_total_epochs(iterations)
                #EightNet.thread = TrainThread(iterations, EightNet.model, EightNet.images, EightNet.labels, EightNet.history)
                #EightNet.thread.start()
                EightNet.model.fit(EightNet.images, EightNet.labels, epochs=iterations, callbacks=[EightNet.history])

    def get_number(image):
        with graph.as_default():
            output = EightNet.model.predict(np.array(image))
            return np.argmax(output)

    def get_training_progress():
        return (EightNet.history.epochs, EightNet.history.total_epochs)

        
        
        