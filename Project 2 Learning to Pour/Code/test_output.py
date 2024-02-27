import numpy as np
import os,random
####*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED']=str(2)
from sklearn.preprocessing import MinMaxScaler
import sys,copy
import tensorflow as tf
from sklearn.metrics import mean_squared_error


random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)
# total arguments
n = len(sys.argv)

def load_dataset(file):
    dataset = np.load(file)

    output = dataset[:, :, 1:2]
    input = np.delete(dataset, 1, axis=2)

    return input, output


filename = sys.argv[1]
input, output = load_dataset(os.path.join("./data",filename))

#print(input.shape)

prediction = np.load('shahabaz_out.npy')
#prediction = np.load('out_1.npy')

#print(prediction.shape)

tot = 0
cnt = 0
for i in range(len(prediction)):
    for j in range(len(prediction[0])):
        if prediction[i][j][0]!=0:
            tot += (prediction[i][j][0] - output[i][j][0]) ** 2
            cnt += 1
print((tot / cnt)**0.5)




