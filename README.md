# DSAI HW2 : Adder-Subtractor-Practice
##### report鏈結：https://nbviewer.jupyter.org/gist/chunyingg/01cf39e01d9ae0b44d65d726803fad35

### 步驟一：設定參數
* 定義訓練集大小以及加入數字
* 分別設立正號跟負號個別的chars
* 給定RNN所需要的參數size

### 步驟二：產生資料
* 產生訓練集以及測試集
* question設定為方程式A+B or A-B , expected則為相對應的答案
* 將資料轉化為one-hot representation¶

### 步驟三：資料處理
#### 設定RNN model
* activation為softmax
* loss function為categorical_crossentropy
* optimizer為adam
* metrics為accuracy

### 步驟四：訓練模型
* 加法模型 (training size = 18000時可獲得最佳validation accuracy)
* 減法模型 (training size = 45000時可獲得最佳validation accuracy)
* 加減法模型(training size = 80000時可獲得最佳validation accuracy)

### 步驟五：驗證模型
