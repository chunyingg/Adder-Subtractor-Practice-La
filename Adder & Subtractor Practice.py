#!/usr/bin/env python
# coding: utf-8

# # DSAI HW2 : Addar & Subtractor Practice




from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range


# # Parameters Config

# ## 設定參數
# ### 定義訓練集大小以及加入數字，分別設立正號跟負號個別的chars
# ### 給定RNN所需要的參數size




class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'





TRAINING_SIZE = 80000
DIGITS = 3
REVERSE = False
MAXLEN = DIGITS + 1 + DIGITS
chars = '0123456789+ '
chars2 = '0123456789- '
chars3 = '0123456789+- '
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1





class CharacterTable(object):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
    
    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x
    
    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[i] for i in x)





ctable = CharacterTable(chars)
ctable2 = CharacterTable(chars2)
ctable3 = CharacterTable(chars3)




print(ctable3.indices_char)


# # ----------------------------加法器---------------------------------

# # Data Generation (adder)

# ### 產生訓練集以及測試集，question設定為方程式A+B，expected則為相對應的答案
# ### 並將資料轉化為one-hot representation




questions = []
expected = []
seen = set()
print('Generating data...')
while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)
    ans += ' ' * (DIGITS + 1 - len(ans))
    if REVERSE:
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))





print(questions[:5], expected[:5])


# # Processing

# ### training size 18000 ， validation size 2000，test size 60000




print('Vectorization...')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(expected), DIGITS + 1, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)





indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# train_test_split
train_x = x[:20000]
train_y = y[:20000]
test_x = x[20000:]
test_y = y[20000:]

split_at = len(train_x) - len(train_x) // 10
(x_train, x_val) = train_x[:split_at], train_x[split_at:]
(y_train, y_val) = train_y[:split_at], train_y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

print('Testing Data:')
print(test_x.shape)
print(test_y.shape)





print("input: ", x_train[:3], '\n\n', "label: ", y_train[:3])


# # Build Model

# ### 設定RNN model
# * activation為softmax
# * loss function為categorical_crossentropy
# * optimizer為adam
# * metrics為accuracy




print('Build model...')

model = Sequential()
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
model.add(layers.RepeatVector(DIGITS + 1))
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

model.add(layers.TimeDistributed(layers.Dense(len(chars), activation='softmax')))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


# # Training




for iteration in range(100):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
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


# # Testing

# ### 訓練好的模型拿來預測測試集，可以看到隨機抽取的預測結果基本上都是對的




print("MSG : Prediction")





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


# # ------------------------------減法器---------------------------------

# # Data Generation (subtractor)

# ### 生成減法的資料，把加號改成減號即可



questions1 = []
expected1 = []
seen1 = set()
print('Generating data...')
while len(questions1) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    key = tuple(sorted((a, b)))
    if key in seen1:
        continue
    seen1.add(key)
    q = '{}-{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a - b)
    ans += ' ' * (DIGITS + 1 - len(ans))
    #if REVERSE:
    #    query = query[::-1]
    questions1.append(query)
    expected1.append(ans)
print('Total addition questions:', len(questions1))




print(questions1[:5], expected1[:5])


# # Processing

# ## 經測試後發現減法器在其他條件不變下需要更大的資料量才能提升準確率，
# ## 因此將訓練集數量增加到45000筆




print('Vectorization...')
x1 = np.zeros((len(questions1), MAXLEN, len(chars2)), dtype=np.bool)
y1 = np.zeros((len(expected1), DIGITS + 1, len(chars2)), dtype=np.bool)
for i, sentence in enumerate(questions1):
    x1[i] = ctable2.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected1):
    y1[i] = ctable2.encode(sentence, DIGITS + 1)





indices1 = np.arange(len(y1))
np.random.shuffle(indices1)
x1 = x1[indices1]
y1 = y1[indices1]

# train_test_split
train_x1 = x1[:50000]
train_y1 = y1[:50000]
test_x1 = x1[50000:]
test_y1 = y1[50000:]

split_at = len(train_x1) - len(train_x1) // 10
(x_train1, x_val1) = train_x1[:split_at], train_x1[split_at:]
(y_train1, y_val1) = train_y1[:split_at], train_y1[split_at:]

print('Training Data:')
print(x_train1.shape)
print(y_train1.shape)

print('Validation Data:')
print(x_val1.shape)
print(y_val1.shape)

print('Testing Data:')
print(test_x1.shape)
print(test_y1.shape)


# # Build Model




print('Build model...')

model1 = Sequential()
model1.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars2))))
model1.add(layers.RepeatVector(DIGITS + 1))

for _ in range(LAYERS):
    model1.add(RNN(HIDDEN_SIZE, return_sequences=True))

model1.add(layers.TimeDistributed(layers.Dense(len(chars2), activation='softmax')))
model1.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model1.summary()


# # Training




for iteration in range(100):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model1.fit(x_train1, y_train1,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val1, y_val1))
    for i in range(10):
        ind = np.random.randint(0, len(x_val1))
        rowx, rowy = x_val1[np.array([ind])], y_val1[np.array([ind])]
        preds = model1.predict_classes(rowx, verbose=0)
        q = ctable2.decode(rowx[0])
        correct = ctable2.decode(rowy[0])
        guess = ctable2.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)


# # Testing




preds = model1.predict_classes(test_x1)

count = 0
for i in range(len(preds)):
    
    q = ctable2.decode(test_x1[i])
    correct = ctable2.decode(test_y1[i])
    guess = ctable2.decode(preds[i], calc_argmax=False)
    
    print('Q', q, end=' ')
    print('T', correct, end=' ')
    if correct == guess:
        print(colors.ok + '☑' + colors.close, end=' ')
        count += 1
    else:
        print(colors.fail + '☒' + colors.close, end=' ')
    print(guess)
    
print(count/len(preds))


# ## 以測試集測試減法器準確率可以達到96.9%




print('the accuracy :')
print(count/len(preds))


# # ---------------------------加法+減法------------------------------

# ### 合併加法跟減法的資料，訓練一個既可以學習加法也可以學習減法的運算模型
# ### 而因為資料變異較大，需要用更大量的資料來訓練，因此訓練集給了80000筆資料




questions = []
expected = []
seen = set()
print('Generating data...')
while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)
    ans += ' ' * (DIGITS + 1 - len(ans))
    questions.append(query)
    expected.append(ans)

print('Total addition questions:', len(questions))





while len(questions) < 160000:
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    q = '{}-{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a - b)
    ans += ' ' * (DIGITS + 1 - len(ans))
    #if REVERSE:
        #query = query[::-1]
    questions.append(query)
    expected.append(ans)





print(questions[:5],expected[:5])
print(questions[159995:],expected[159995:])





print('Vectorization...')
x_ = np.zeros((len(questions), MAXLEN, len(chars3)), dtype=np.bool)
y_ = np.zeros((len(expected), DIGITS + 1, len(chars3)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x_[i] = ctable3.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y_[i] = ctable3.encode(sentence, DIGITS + 1)





indices_ = np.arange(len(y_))
np.random.shuffle(indices_)
x_ = x_[indices_]
y_ = y_[indices_]

# train_test_split
train_x_ = np.concatenate((x_[:40000],x_[120000:]),axis=0)
train_y_ = np.concatenate((y_[:40000],y_[120000:]),axis=0)
test_x_ = x_[40001:120001]
test_y_ = y_[40001:120001]

split_at = len(train_x_) - len(train_x_) // 10
(x_train_, x_val_) = train_x_[:split_at], train_x_[split_at:]
(y_train_, y_val_) = train_y_[:split_at], train_y_[split_at:]

print('Training Data:')
print(x_train_.shape)
print(y_train_.shape)

print('Validation Data:')
print(x_val_.shape)
print(y_val_.shape)

print('Testing Data:')
print(test_x_.shape)
print(test_y_.shape)





print('Build model...')

model11 = Sequential()
model11.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars3))))
model11.add(layers.RepeatVector(DIGITS + 1))

for _ in range(LAYERS):
    model11.add(RNN(HIDDEN_SIZE, return_sequences=True))

model11.add(layers.TimeDistributed(layers.Dense(len(chars3), activation='softmax')))
model11.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model11.summary()





for iteration in range(100):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model11.fit(x_train_, y_train_,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val_, y_val_))
    for i in range(10):
        ind = np.random.randint(0, len(x_val_))
        rowx, rowy = x_val_[np.array([ind])], y_val_[np.array([ind])]
        preds = model11.predict_classes(rowx, verbose=0)
        q = ctable3.decode(rowx[0])
        correct = ctable3.decode(rowy[0])
        guess = ctable3.decode(preds[0], calc_argmax=False)
        print('Q', q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)





preds_ = model11.predict_classes(test_x_)

count_ = 0
for i in range(len(preds_)):
    
    q = ctable3.decode(test_x_[i])
    correct = ctable3.decode(test_y_[i])
    guess = ctable3.decode(preds_[i], calc_argmax=False)
    
    print('Q', q, end=' ')
    print('T', correct, end=' ')
    if correct == guess:
        print(colors.ok + '☑' + colors.close, end=' ')
        count_ += 1
    else:
        print(colors.fail + '☒' + colors.close, end=' ')
    print(guess)
    
print(count_/len(preds_))


# ## 測試後準確率可以達到96.8%




print('the accuracy :')
print(count_/len(preds_))







