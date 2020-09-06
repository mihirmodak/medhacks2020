import time
from flask import Flask, Response
import os
import numpy as np
from os import listdir
from os.path import isfile, join
import scipy.io as sio
import pandas as pd
from tensorflow.keras.models import load_model
import argparse
from tqdm import tqdm
import statistics
from statistics import mode
import matplotlib.pyplot as plt
import io
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "hello_world"

@app.route('/time')
def get_current_time():
    return {'time':time.time()}

@app.route('/ecg')
def plot_graph():
    fig = create_graph()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def repeat_preds(num_iters=4):
    # np.random.seed(7)

    number_of_classes = 4 #Total number of classes

    mypath = 'testing/data/' #Testing directory
    onlyfiles = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f[0] == 'A')]
    bats = [f for f in onlyfiles if f[7] == 'm']
    check = 100
    mats = [f for f in bats if (np.shape(sio.loadmat(mypath + f)['val'])[1] >= check)]
    size = len(mats)
    print('Total training size is ', size)
    big = 10100
    X = np.zeros((size, big))
    ######Old stuff
    # for i in range(size):
        # X[i, :] = sio.loadmat(mypath + mats[i])['val'][0, :check]
    ######

    for i in range(size):
        dummy = sio.loadmat(mypath + mats[i])['val'][0, :]
        if (big - len(dummy)) <= 0:
            X[i, :] = dummy[0:big]
        else:
            b = dummy[0:(big - len(dummy))]
            goal = np.hstack((dummy, b))
            while len(goal) != big:
                b = dummy[0:(big - len(goal))]
                goal = np.hstack((goal, b))
            X[i, :] = goal

    target_train = np.zeros((size, 1))
    Train_data = pd.read_csv(mypath + 'REFERENCE.csv', sep=',', header=None, names=None)
    for i in range(size):
        if Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'N':
            target_train[i] = 0
        elif Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'A':
            target_train[i] = 1
        elif Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'O':
            target_train[i] = 2
        else:
            target_train[i] = 3

    Label_set = np.zeros((size, number_of_classes))
    for i in range(size):
        dummy = np.zeros((number_of_classes))
        dummy[int(target_train[i])] = 1
        Label_set[i, :] = dummy

    X = (X - X.mean())/(X.std()) #Some normalization here
    X = np.expand_dims(X, axis=2) #For Keras's data input size

    values = [i for i in range(size)]
    permutations = np.random.permutation(values)
    X = X[permutations, :]
    Label_set = Label_set[permutations, :]

    """
    X is the data, Label_set are the correct labels for X
    """

    model = load_model("./Conv_models/BestModel1D13.h5")
    model.reset_metrics()
    # score = model.evaluate(X, Label_set, verbose=1)
    # print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    index = np.random.randint(len(X))
    X_single = X[index-1 if index != 0 else index: index+18 if index != 0 else index+19]

    Train_data.set_index(0, inplace=True)
    decoder = {0:'N', 1:'A', 2:'O', 3:'~'}
    final_preds_list=[]

    for i in range(len(X_single)):
        temp_list = []
        for _ in range(num_iters):
            print(i)
            # print(X_single[i:i+1])
            preds = model.predict(X_single[i:i+1], verbose = 1)

            preds = preds.tolist()[0]
            # print(preds)
            # print(preds.index(max(preds)))

            category = decoder[ preds.index(max(preds)) ]

            temp_list.append(category if category != 'A' else 'A')

        final_preds_list.append(mode(temp_list) if temp_list.count('A') < 3 or ['A','A'] not in temp_list else 'A' )
    try:
    	final_pred = [ mode(final_preds_list) if final_preds_list.count('A') < 3 or ['A','A'] not in final_preds_list else 'A' ][0]
    except statistics.StatisticsError:
    	final_pred = max(set(lst), key=lst.count)
    print("Prediction={}, from {}".format(final_pred, final_preds_list))

    return final_pred, X_single

def create_graph():
    final_pred, X_single = repeat_preds()
    fig = Figure()
    ax = fig.add_subplot(1,1,1)

    raw_data = X_single[:]
    raw_data = np.array(raw_data)
    
    one_d_data = np.ravel(raw_data)
    time_vals = range(len(one_d_data))

    ax.plot(time_vals, one_d_data)
#   ax.xlim(0, 20000)
#   ax.xticks(range(0,20000,2500))
#   ax.autoscale(enable=True, axis='y', tight=True)

    return fig

if __name__=='__main__':
    # ap = argparse.ArgumentParser()
    # args = vars(ap.parse_args())
    app.run(host='0.0.0.0', debug=True, port=3000)
