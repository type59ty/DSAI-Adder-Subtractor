from keras.models import Sequential, load_model
from keras.layers import SimpleRNN, Activation, Dense
from keras import layers
import numpy as np
from six.moves import range
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import csv
import os
import os.path
import argparse
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

class CharacterTable(object):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c,i) for i,c in enumerate(self.chars))
        self.indices_char = dict((i,c) for i,c in enumerate(self.chars))
    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i,c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x
    
    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[i] for i in x)


def gen_data(TRAINING_SIZE, DIGITS, chars, ctable, cal_type):
    questions = []
    expected  = []
    seen = set()
    print('Generating data...')
    while len(questions) < TRAINING_SIZE:
        f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS+1))))
        a, b = f(), f()
        if DIGITS > 2:
            key  = tuple(sorted((a,b)))
            if key in seen:
                continue
            seen.add(key)
        if cal_type == 'add':
            q = '{}+{}'.format(a,b)
            query = q + ' ' * (MAXLEN - len(q))
            ans = str(a+b)
            ans += ' ' * (DIGITS + 1 - len(ans))
        elif cal_type == 'sub':
            q = '{}-{}'.format(a,b)
            query = q + ' ' * (MAXLEN - len(q))
            ans = str(a-b)
            ans += ' ' * (DIGITS + 1 - len(ans))
        elif cal_type == 'add_sub':
            choose_add = np.random.randint(2,size=1)
            if choose_add:
                q = '{}+{}'.format(a,b)
                query = q + ' ' * (MAXLEN - len(q))
                ans = str(a+b)
            else: 
                q = '{}-{}'.format(a,b)
                query = q + ' ' * (MAXLEN - len(q))
                ans = str(a-b)
            ans += ' ' * (DIGITS + 1 - len(ans))
        else:
            q = '{}*{}'.format(a,b)
            query = q + ' ' * (MAXLEN - len(q))
            ans = str(a*b)
            ans += ' ' * (DIGITS + 3 - len(ans))
        if REVERSE:
            qurey = query[::-1]
        questions.append(query)
        expected.append(ans)
    print('Vectorization...')
    x = np.zeros((len(questions), MAXLEN,   len(chars)), dtype=np.bool)
    if cal_type == 'mul':
        y = np.zeros((len(expected),  DIGITS+3, len(chars)), dtype=np.bool)
    else:
        y = np.zeros((len(expected),  DIGITS+1, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, MAXLEN)
    if cal_type == 'mul':
        for i, sentence in enumerate(expected):
            y[i] = ctable.encode(sentence, DIGITS+3)
    else:
        for i, sentence in enumerate(expected):
            y[i] = ctable.encode(sentence, DIGITS+1)
    
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    
    # train_test_split
    TrS = TRAINING_SIZE-10000
    train_x = x[:TrS]
    train_y = y[:TrS]
    test_x  = x[TrS:]
    test_y  = y[TrS:]
     
    split_at = len(train_x) - len(train_x) // 10
    (x_train, x_val) = train_x[:split_at], train_x[split_at:]
    (y_train, y_val) = train_y[:split_at], train_y[split_at:]
    print('Generated Data: ') 
    print('Training Data:')
    print(x_train.shape)
    print(y_train.shape)
    print('Validation Data:')
    print(x_val.shape)
    print(y_val.shape)
    print('Testing Data:')
    print(test_x.shape)
    print(test_y.shape)
    data = [x_train,x_val,y_train,y_val,test_x,test_y]
    return data


def build(DIGITS, chars, HIDDEN_SIZE, LAYERS, cal_type):
    LSTM = layers.LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(
                HIDDEN_SIZE, 
                input_shape=(MAXLEN, len(chars))
    ))
    if cal_type=='mul':
        model.add(layers.RepeatVector(DIGITS + 3))
    else:
        model.add(layers.RepeatVector(DIGITS + 1))
    for _ in range(LAYERS):
        model.add(LSTM(HIDDEN_SIZE, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(len(chars), activation='softmax')))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

def train(model, BATCH_SIZE, x_train, y_train, x_val, y_val, test_x, test_y, ITER, cal_type, DIGITS):
    with open('output/out_{}_{}d.csv'.format(cal_type,DIGITS), 'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['iter','tr_loss','tr_acc','val_loss','val_acc'])
    tr_acc_list=[]
    te_acc_list=[]
    for iteration in range(1,ITER+1):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        hist = model.fit(x_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=1,
                  validation_data=(x_val, y_val))
        for i in range(10):
            ind = np.random.randint(0, len(x_val))
            rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
            preds = model.predict_classes(rowx, verbose=0)
            q = ctable.decode(rowx[0])
            correct = ctable.decode(rowy[0])
            guess = ctable.decode(preds[0], calc_argmax=False)
            print('Q', q[::-1] if REVERSE else q, end=' ')
            print('T', correct, end=' ')
            if correct == guess:
                print(colors.ok + '☑' + colors.close, end=' ')
            else:
                print(colors.fail + '☒' + colors.close, end=' ')
            print(guess)
        if iteration%5 == 0 or iteration==1:
            tr_acc   = hist.history.get('acc')[-1]
            tr_loss  = hist.history.get('loss')[-1]
            val_acc  = hist.history.get('val_acc')[-1]
            val_loss = hist.history.get('val_loss')[-1]
            tr_acc_list.append(tr_acc)
            _, te_acc = model.evaluate(test_x, test_y)
            te_acc_list.append(te_acc)
            with open('output/out_{}_{}d.csv'.format(cal_type,DIGITS), 'a',newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([iteration, '{:.4f}'.format(tr_loss),  '{:.4f}'.format(tr_acc),\
                                            '{:.4f}'.format(val_loss), '{:.4f}'.format(val_acc)])
    model.save('model/model_{}_{}d.h5'.format(cal_type,DIGITS))
    return model, tr_acc_list, te_acc_list

def show(model, test_x, test_y):
    print("Testing result:")
    for i in range(10):
        ind = np.random.randint(0, len(test_x))
        rowx, rowy = test_x[np.array([ind])], test_y[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)


class myParser(argparse.ArgumentParser):
    def format_help(self):
        help()
        return

def help():
    print('Usage: python main.py [-t CAL_TYPE] [-d DIGITS] [-m MODEL_TYPE]')
    print('calculation type options: (default: -t add)')
    print(' -t add                 addition')
    print(' -t sub                 subtraction')
    print(' -t add_sub             addition & subtraction')
    print(' -t mul                 multiplication')
    print('number of digits options: (default: -d 3)')
    print(' -d 2                   two   digits')
    print(' -d 3                   three digits')
    print(' -d 4                   four  digits')
    print('model type options: (default: -t none)')
    print(' -m add                 addition model')
    print(' -m sub                 subtraction model')
    print(' -m add_sub             addition & substraction model')
    print(' -m mul                 multiplication model')
    print()


if __name__=='__main__':
    help()
    parser   = myParser()
    parser.add_argument('-t', default='add' ,help='input calculation type')
    parser.add_argument('-d', default='3'   ,help='input number of digits')
    parser.add_argument('-m', default='none',help='load pre-train model')
    args     = parser.parse_args()
    cal_type = args.t
    DIGITS   = int(args.d)
    mpath = 'model/model_{}_{}d.h5'.format(args.m,DIGITS)
    TRAINING_SIZE = 30000
    HIDDEN_SIZE   = 128
    BATCH_SIZE    = 128
    LAYERS = 1
    ITER   = 100
    REVERSE = False
    MAXLEN  = DIGITS + 1 + DIGITS
    chars = '0123456789+-* '
    ctable = CharacterTable(chars)
    data = gen_data(TRAINING_SIZE, DIGITS, chars, ctable, cal_type)
    
    x_train ,x_val, y_train, y_val, test_x, test_y = data[0],data[1],data[2],data[3],data[4],data[5]
    # load exists model
    if os.path.exists(mpath):
        print("Loading the model...")
        print()
        model = load_model(mpath)
        model.summary()
        score_te = model.evaluate(test_x, test_y, batch_size = BATCH_SIZE, verbose=False)
        score_tr = model.evaluate(x_train, y_train, batch_size = BATCH_SIZE, verbose=False)
        show(model, test_x, test_y)
        print("Training acc: {:.5f}, Training loss: {:.4f}".format(score_tr[1], score_tr[0]))
        print("Testing  acc: {:.5f}, Testing  loss: {:.4f}".format(score_te[1], score_te[0]))
    # retrain the model
    else:
        model = build(DIGITS, chars, HIDDEN_SIZE, LAYERS, cal_type)
        print("Start training...")
        model, tr_acc_list, te_acc_list= train(model, BATCH_SIZE, x_train, y_train, x_val, y_val, \
                                                                     test_x, test_y, ITER, cal_type, DIGITS)
        score_te = model.evaluate(test_x, test_y, verbose=False)
        score_tr = model.evaluate(x_train, y_train, verbose=False)
        with open('output/out_{}_{}d.csv'.format(cal_type, DIGITS), 'a',newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['test_loss','test_acc'])
            writer.writerow(['{:.4f}'.format(score_te[0]),'{:.4f}'.format(score_te[1])])
        print("Training acc: {:.4f}, Training loss: {:.4f}".format(score_tr[1], score_tr[0]))
        print("Testing  acc: {:.4f}, Testing  loss: {:.4f}".format(score_te[1], score_te[0]))
        x = np.arange(len(tr_acc_list))
        plt.plot(x, tr_acc_list, label='train acc')
        plt.plot(x, te_acc_list, label='test  acc',linestyle='--')
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.xlim(0, ITER/5)
        plt.xticks(range(len(tr_acc_list)))
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.savefig('img/acc_{}_{}d.png'.format(cal_type,DIGITS))
        # plt.show()



