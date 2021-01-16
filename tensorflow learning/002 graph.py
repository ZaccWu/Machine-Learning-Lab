import tensorflow as tf

two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node
sess = tf.Session()

print(sum_node)                 # Tensor("add:0", shape=(), dtype=int32)
print(type(sum_node))           # <class 'tensorflow.python.framework.ops.Tensor'>
print(sess.run(sum_node))       # 5
print(type(sess.run(sum_node))) # <class 'numpy.int32'>