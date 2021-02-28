'''
这个例子为了说明numpy和tensorflow使用方式的相互转换
以HSIC的计算为例，这两种方式算出的结果是相同的
'''

import numpy as np
import tensorflow as tf

def hsic(Kx, Ky):
    Kxy = np.dot(Kx, Ky)
    n = Kxy.shape[0]
    h = np.trace(Kxy) / n** 2+ np.mean(Kx) * np.mean(Ky) - 2* np.mean(Kxy) / n
    print(np.trace(Kxy), np.mean(Kx), np.mean(Ky), np.mean(Kxy), n)
    return h * n** 2/ (n - 1)** 2

x = np.random.randn(1000)
y = np.random.randn(1000)
Kx = np.expand_dims(x, 0) - np.expand_dims(x, 1)
Kx = np.exp(- Kx** 2) # 计算核矩阵
Ky = np.expand_dims(y, 0) - np.expand_dims(y, 1)
Ky = np.exp(- Ky** 2) # 计算核矩阵

# print(x.shape)  # (1000,)
# print(np.expand_dims(x, 0).shape)   # (1,1000)
# print(np.expand_dims(x, 1).shape)   # (1000,1)
# print(Kx.shape) # (1000,1000)
print(hsic(Kx, Ky)) # 计算HSIC

def thsic(tKx, tKy):
    tKxy = tf.matmul(tKx, tKy)
    n = int(tKxy.shape[0])
    with tf.Session() as sess:
        tracetKxy = sess.run(tf.trace(tKxy))
        meantKx = sess.run(tf.reduce_mean(tKx))
        meantKy = sess.run(tf.reduce_mean(tKy))
        meantKxy = sess.run(tf.reduce_mean(tKxy))
    h = tracetKxy / n**2 + meantKx*meantKy -2* meantKxy/ n
    print(tracetKxy, meantKx, meantKy, meantKxy, n)
    return h *n**2/(n-1)**2

tx = tf.convert_to_tensor(x)
ty = tf.convert_to_tensor(y)
tKx = tf.expand_dims(tx, 0) - tf.expand_dims(tx, 1)
tKx = tf.exp(-tf.square(tKx))
tKy = tf.expand_dims(ty, 0) - tf.expand_dims(ty, 1)
tKy = tf.exp(-tf.square(tKy))

print(thsic(tKx, tKy))
