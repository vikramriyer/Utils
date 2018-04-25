import numpy as np
import pandas as pd
from yaml import load, dump
from PIL import Image
import matplotlib.pyplot as plt

with open("/home/ubuntu/config/config.yaml", "r") as f:
        config = load(f)

PATH = "/home/ubuntu/data/mnist/"
filename =  PATH + '/' + 'train.csv'
data = np.genfromtxt(filename, delimiter=",", skip_header=1)

print(data.shape)

N = 42000
# data = np.arange(N*785).reshape(-1, 785)
np.random.shuffle(data)

train = data[:int(N*0.8)]
validate = data[int(N*0.8):]

print(train.shape)
print(validate.shape)
train_labels = train[:,0]
validate_labels = validate[:,0]

def get_label(number):
    number = int(number)
    return config[number]

def save_images(data, option):
    count = 1
    for row in data:
        plt.imsave("/home/ubuntu/data/mnist/" + option + "/" + get_label(row[0]) + "/" + str(count) +
                   ".png", np.reshape(row[1:785], (28,28)))
        count += 1

save_images(train, "train")
save_images(validate, "validate")

