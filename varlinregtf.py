import numpy as np
import tensorflow as tf

class VarLinRegTf:
    def train(self, Xtrain, ytrain):
        # Data properties
        M = Xtrain.shape[1]  # num input dimensions

        # Initializations
        a0 = tf.constant(1.0)  # hyperparameters
        b0 = tf.constant(1.0)
        beta = tf.constant(1.0)  # noise precision
        max_iter = 10  # maximum number of iterations allowed

        # tf Graph Input
        x = tf.placeholder(tf.float32,shape=Xtrain.shape,name="x")
        y = tf.placeholder(tf.float32,shape=ytrain.shape,name="y")

        initialInputMu = np.random.random([M,])
        initialInputSigma = np.random.random([M, M])

        m = tf.Variable(initialInputMu, name="mu")
        S = tf.Variable(initialInputSigma, name="sigma")
        S = tf.matmul(S, S, transpose_b=True)
        EwTw = tf.add(tf.reduce_sum(tf.multiply(m, m)), tf.trace(S))
        EwTw = tf.cast(EwTw, tf.float32)

        # Update q(alpha | aN, bN)
        aN = tf.add(a0, tf.divide(M, tf.constant(2.0)))
        bN = tf.add(b0, tf.multiply(tf.constant(0.5), EwTw))
        Ealpha = tf.divide(aN, bN)

        identity = tf.Variable(initial_value = np.identity(M), dtype="float32")

        S = tf.matrix_inverse(tf.multiply(Ealpha, identity) + tf.multiply(beta, tf.matmul(x, x, transpose_a=True)))
        m = tf.multiply(beta, tf.matmul(tf.matmul(S, x, transpose_b=True), tf.expand_dims(y, 1)))
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            for iter in range(max_iter):
                learntM, learntS = sess.run([m, S], feed_dict={x: Xtrain, y: ytrain})

            self.beta = beta
            self.m = learntM.T
            self.S = learntS
            self.Ealpha = Ealpha

            self.m = np.reshape(self.m, (13,))

    # Prediction (posterior predictive calculation) function
    def predict(self, Xtest):
        # Predictive mean (compare this to vanilla linear regression)
        pmean = Xtest.dot(self.m)
        # Predictive variance (does this exist in vanilla linear regression?)
        pvar = 1.0 / self.beta + Xtest.dot(self.S).dot(Xtest.T).trace()

        return (pmean, pvar)

