import math
import tensorflow as tf

sess = tf.Session()

# Here we define two variables, a and b (just as f(X) = aX+b), with initial value 1.
# 1. The optimizer with adjust these two values to make the algorithm more accurate
a = tf.Variable(tf.constant(1.))
b = tf.Variable(tf.constant(1.))

# 2. This is the Input data and the placeholder (like the parameter in a Python function)
input = 5.
# It will be used in the definition of the algorithm
input_placeholder = tf.placeholder(dtype=tf.float32)

# 3. This is the definition of the algorithm, we define a simple algorithm f(X)=aX+b here
algo = tf.add(tf.multiply(a, input_placeholder), b)

# 4. This is the 'outcome' data and its placeholder
out = 50.
out_placeholder = tf.placeholder(dtype=tf.float32)
# 5. This is loss function
loss = tf.square(tf.subtract(algo, out_placeholder))

# 6. This is the optimizer
# The optimizer need to 'call' the loss function
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# Start training (for 50 times)
init = tf.global_variables_initializer()
sess.run(init)
for i in range(50):
    # the 'real data' will fill the placeholder when training, feed_dict={input_placeholder:input,out_placeholder:out}
    sess.run(train, feed_dict={input_placeholder: input, out_placeholder: out})
    a_val, b_val = (sess.run(a), sess.run(b))
    result = sess.run(algo, feed_dict={input_placeholder: input})

    if i % 5 == 0:
        print(str(a_val)+' * '+str(input)+' + '+str(b_val)+' = '+str(result))
    if math.isclose(result, 50):
        break
