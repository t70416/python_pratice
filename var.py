import tensorflow as tf

state = tf.Variable(0, name = 'counter') #設定變數 tf.Variable(初始值, name = '命名')
#print(state.name)

one = tf.constant(1) #常數

new_value = tf.add(state , one)
update = tf.assign(state, new_value) #state = new_value

init = tf.global_variables_initializer() #初始變數(must)

with tf.Session() as sess :
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
