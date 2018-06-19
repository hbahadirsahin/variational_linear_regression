#
# Variational Inference Code for Bayesian Linear Regression
#
# Course material for CS 458
#
# Contact: Melih Kandemir, Ozyegin University, Dept of Computer Science
# E-mail:name [dot] surname [at] ozyegin [dot] edu [dot] tr
#
# No commercial use allowed
# All rights reserved
#
import numpy as np

class VarLinReg:

  # Training (posterior inference) function
  def train(self, Xtrain, ytrain):

      # Data properties
      #N = Xtrain.shape[0] # num training points
      M = Xtrain.shape[1] # num input dimensions

      # Initializations      
      a0 = 1.0 # hyperparameters
      b0 = 1.0
      beta = 1.0 # noise precision
      
      max_iter = 10 # maximum number of iterations allowed
            
      # Initialize q(w | m, S)
      m = np.random.random([M,])
      S = np.random.random([M, M])
      S = S.dot(S.T) # Covariance matrix has to be symmetric and PSD
      EwTw = m.dot(m) + S.trace() # Bishop, 10.104, pp 488 (the denominator!)

      # Update q(alpha | aN, bN)
      aN = a0 + M / 2.0  # Bishop, 10.94, pp 487
      bN = b0 + 0.5 * EwTw  # Bishop, 10.95, pp 487
      Ealpha = aN / bN  # Bishop, 10.102, pp 488
      # Iterations
      for mm in range(max_iter):
          # Update q(w | m, S)
          S = np.linalg.inv(Ealpha * np.identity(M) + beta*Xtrain.T.dot(Xtrain))
          m = beta*S.dot(Xtrain.T).dot(ytrain)

      # Store the state
      self.beta = beta
      self.m = m
      self.S = S
      self.Ealpha = Ealpha

  # Prediction (posterior predictive calculation) function
  def predict(self, Xtest):
      # Predictive mean (compare this to vanilla linear regression)
      pmean = Xtest.dot(self.m)
      # Predictive variance (does this exist in vanilla linear regression?)
      pvar  = 1.0 / self.beta + Xtest.dot(self.S).dot(Xtest.T).trace()

      return (pmean, pvar)