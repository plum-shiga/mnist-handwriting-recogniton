from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.vis_utils import plot_model
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# (◯, 28, 28)のデータを(◯, 784)に次元を減らす。(簡略化のため)
shapes = X_train.shape[1] * X_train.shape[1]

# 0ばっかりの二次元配列
X_train = X_train.reshape(X_train.shape[0], shapes)[:6000]
# 0ばっかりの二次元配列
X_test = X_test.reshape(X_test.shape[0], shapes)[:1000]

y_train = to_categorical(y_train)[:6000] # 0, 1ばっかりの二次元配列
y_test = to_categorical(y_test)[:1000] # 0, 1ばっかりの二次元配列

# modelはインスタンス。実態はバイナリ
model = Sequential()

# 入力ユニット数は784, 1つ目の全結合層の出力ユニット数は256
# Denseは全結合相を出すため, でそれをmodelにaddしている！
model.add(Dense(256, input_dim=784))
model.add(Activation("sigmoid")) # sigmoidで活性化させる

# 2つ目の全結合層の出力ユニット数は128。活性化関数はrelu。
model.add(Dense(128))
model.add(Activation("relu"))


# 3つ目の全結合層（出力層）の出力ユニット数は10
model.add(Dense(10))
model.add(Activation("softmax"))

# 学習処理を設定して、モデルの生成が終了する
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
# 学習データではなく、テストデータを使って新規データに対する精度を測る
score = model.evaluate(X_test, y_test, verbose=1)

# モデル構造の出力
plot_model(model, "model125.png", show_layer_names=False)
# モデル構造の可視化
image = plt.imread("model125.png")
plt.figure(dpi=150)
plt.imshow(image)
plt.show()
