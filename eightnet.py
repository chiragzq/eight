import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import struct
import threading
import datetime
import os

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
        self.model.fit(self.images, self.labels, epochs=self.iterations, callbacks=[self.callback])
"""
graph = tf.get_default_graph()
class EightNet:
    def __init__(self):
        self.labels = None
        self.images = None
        self.initialLearningRate = 0.06
        self.model = None
        self.history = TrainingProgressHistory()
        self.thread = None
        self.prev_acc = -1
    
    def load_data(self, num_images, labelPath="train-labels-idx1-ubyte", imagePath="train-images-idx3-ubyte", testImagePath="t10k-images-idx3-ubyte", testLabelPath="t10k-labels-idx1-ubyte"):
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
                    print("read %d images\r" % (i + 1), end="\r")
        print("Loaded %d/%d images" % (min(num_images, num_items), num_items))
        for i in range(0, len(tempLabels)):
            arr = np.zeros(10)
            arr[tempLabels[i]] = 1
            tempLabels[i] = arr
        self.labels = np.array(tempLabels)
        self.images = np.array(tempImages) / 255.0
        
        tempLabels = []
        tempImages = []
        with open(testLabelPath, "rb") as f:
            magic = struct.unpack('>i', f.read(4))[0]
            num_items = struct.unpack(">i", f.read(4))[0]
            for i in range(0, num_items):
                tempLabels.append(struct.unpack("B", f.read(1))[0])
        print("Loaded %d/%d testing labels" % (num_items, num_items))
        with open(testImagePath, "rb") as f:
            magic = struct.unpack('>i', f.read(4))[0]
            num_items = struct.unpack('>i', f.read(4))[0]
            num_rows = struct.unpack('>i', f.read(4))[0]
            num_columns = struct.unpack('>i', f.read(4))[0]
            for i in range(0, num_items):
                tempImages.append([])
                for j in range(0, num_rows):
                    tempImages[i].append([])
                    for k in range(0, num_columns):
                        byte = f.read(1)
                        tempImages[i][j].append(struct.unpack("B", byte)[0])
                if i % 25 == 24:
                    print("read %d testing images\r" % (i + 1), end="\r")

        for i in range(0, len(tempLabels)):
            arr = np.zeros(10)
            arr[tempLabels[i]] = 1
            tempLabels[i] = arr
                #self.print_array(self.images[5])
        #cv2.imshow("image", self.images[5])
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        self.testLabels = np.array(tempLabels)
        self.testImages = np.array(tempImages) / 255.0
        
        
    def initialize_model(self):
        with graph.as_default():
            self.model = keras.Sequential([
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Dense(128, activation=tf.nn.relu),
                keras.layers.Dense(10, activation=tf.nn.softmax)
            ])
            self.model.compile(optimizer=tf.keras.optimizers.SGD(self.initialLearningRate), loss="mean_squared_error", metrics=["accuracy"])
    
    def train_model(self, iterations):
        self.prev_acc = -1
        with graph.as_default():
            if self.thread == None or not self.thread.isAlive():
                self.history.set_total_epochs(iterations)
                #self.thread = TrainThread(iterations, self.model, self.images, self.labels, self.history)
                #self.thread.start()
                self.model.fit(self.images, self.labels, epochs=iterations, callbacks=[self.history])

    def get_number(self, image):
        with graph.as_default():
            output = self.model.predict(np.array(image))
            return np.argmax(output)

    def get_training_progress(self):
        return (self.history.epochs, self.history.total_epochs)

    def test(self):
        correct = 0
        for image, label in zip(self.testImages, self.testLabels):
            number = self.get_number([image])
            if number == np.argmax(label):
                correct += 1;
        self.prev_acc = correct
        return "%d/%d" % (correct, len(self.testImages))
        
    def save_model(self):
        timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H:%M")
        if self.prev_acc == -1:
            self.test()
        acc = "%d-%d" % (self.prev_acc, len(self.testImages))
        self.model.save("models/%s_%s.h5" % (timestamp, acc))
    
    def load_model(self, name):
        with graph.as_default():
            self.model = keras.models.load_model("models/" + name)

        
    def get_models(self):
        return os.listdir("models")
        
    def print_array(self, arr):
        print("[", end="")
        for a in arr:
            print("[", end="")
            for b in a:
                print("%0.5f," % b, end="")
            print("],", end="")
        print("]", end="\n")
        
eightNet = EightNet()