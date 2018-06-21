# !/usr/bin/python

# MCMC Random Walk for Feedforward Neural Network for One-Step-Ahead Chaotic Time Series Prediction

# based on: https://github.com/rohitash-chandra/FNN_TimeSeries
# based on: https://github.com/rohitash-chandra/mcmc-randomwalk

import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math

######################################################
# DEFINE A NEURAL NETWORK CLASS
######################################################
class Network:
    def __init__(self, Topo, Train, Test):
        self.Top = Topo  # NN topology [input, hidden, output]
        self.TrainData = Train
        self.TestData = Test
        np.random.seed()

        self.W1 = np.random.randn(self.Top[0], self.Top[1]) / np.sqrt(self.Top[0])
        self.B1 = np.random.randn(1, self.Top[1]) / np.sqrt(self.Top[1])  # bias first layer
        self.W2 = np.random.randn(self.Top[1], self.Top[2]) / np.sqrt(self.Top[1])
        self.B2 = np.random.randn(1, self.Top[2]) / np.sqrt(self.Top[1])  # bias second layer

        self.hidout = np.zeros((1, self.Top[1]))  # output of first hidden layer
        self.out = np.zeros((1, self.Top[2]))  # output last layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sampleEr(self, actualout):
        error = np.subtract(self.out, actualout)
        sqerror = np.sum(np.square(error)) / self.Top[2]
        return sqerror

    def ForwardPass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer

    def BackwardPass(self, Input, desired, vanilla):
        out_delta = (desired - self.out) * (self.out * (1 - self.out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))

        self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
        self.B2 += (-1 * self.lrate * out_delta)
        self.W1 += (Input.T.dot(hid_delta) * self.lrate)
        self.B1 += (-1 * self.lrate * hid_delta)

    def decode(self, w):
        w_layer1size = self.Top[0] * self.Top[1]
        w_layer2size = self.Top[1] * self.Top[2]

        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))
        self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.Top[1]]
        self.B2 = w[w_layer1size + w_layer2size + self.Top[1]:w_layer1size + w_layer2size + self.Top[1] + self.Top[2]]

    def evaluate_proposal(self, data, w):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.

        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)

        for pat in range(0, size):
            Input[:] = data[pat, 0:self.Top[0]]
            Desired[:] = data[pat, self.Top[0]:]

            self.ForwardPass(Input)
            fx[pat] = self.out

        return fx


#-------------------------------------------------------------------------------
# DEFINE A MARKOV CHAIN MONTE CARLO CLASS - FOR THE NEURAL NETWORK
#-------------------------------------------------------------------------------
class MCMC:
    def __init__(self, samples, traindata, testdata, topology):
        self.samples = samples  # NN topology [input, hidden, output]
        self.topology = topology  # max epocs
        self.traindata = traindata  #
        self.testdata = testdata
        # ----------------

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def reduceData(self, data, incl):
        fltre = incl>0
        return data[fltre]

    def modifyIncludedData(self, incl):
        newincl = incl.copy()
        pos = random.choice(list(range(0, len(incl))))
        if newincl[pos]==0: 
           newincl[pos]=1
        else: 
           newincl[pos]=0
        return newincl

    def log_likelihood(self, neuralnet, data, w, tausq):
        y = data[:, self.topology[0]]
        fx = neuralnet.evaluate_proposal(data, w)
        rmse = self.rmse(fx, y)
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
        return [np.sum(loss), fx, rmse]

    def log_prior(self, sigma_squared, nu_1, nu_2, w, tausq):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        logp = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return logp


    def sampler(self):

        # ------------------- initialize MCMC

        # How may training and test points? 
        # shape[0] Returns the first dimension of the array
        # In this instance it is the number of discrete (x,y) combinations in the data set
        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]

        # Samples is the number of samples we are going to take in the run of MCMC
        samples = self.samples

        # HAWKINS: I am inserting some code here to include a random walk over data points. 
        # WORK-IN-PROGRESS: Not certain yet how to initialise and whether
        # this needs to be factored into the calculate of the likelihood.

        # A vector that indicate whether to include a given data point in the training set
        incl = np.ones(trainsize)
        # incl =  np.array([random.choice([0,1]) for _ in range(trainsize)])

        # History of that vector - This will form the posterior over DATA INCLUSION
        pos_incl = np.ones((samples, trainsize))

        # Initialise a vector with a sequence of values equal to the length of the train and test sets
        # We only do this for plotting - these are the x-coordinates for the plot data
        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)

        # The topology of the neural network # [input, hidden, output]
        netw = self.topology

        # Copy the y values into an independent vector
        y_test = self.testdata[:, netw[0]]
        y_train = self.traindata[:, netw[0]]
        print(y_train.size)
        print(y_test.size)

        # The total number of parameters for the neural network
        # is the number of weights and bias
        w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]  

	# Posterior distribution of all weights and bias over all samples
	# We will take 'samples' number of samples
	# and there will be a total of 'w_size' parameters in the model.
        # We collect this because it will hold the empirical data for our 
        # estimate of the posterior distribution. 
        pos_w = np.ones((samples, w_size))

        # TAU IS THE STANDARD DEVIATION OF THE ERROR IN THE DATA GENERATING FUNCTIONS
        # I.E. WE ASSUME THAT THE MODEL WILL BE TRYING TO LEARN SOME FUNCTION F(X)
	# AND THAT THE OBSERVED VALUES Y = F(X) + E
        # THIS STORES THE POSTERIOR DISTRIBUTION ACROSS THESE WEIGHTS AS GENERATED BY THE MCMC PROCESS 
        pos_tau = np.ones((samples, 1))

	# F(X) BUFFER - ALL NETWORK OUTPUTS WILL BE STORED HERE
        fxtrain_samples = np.ones((samples, trainsize))  # fx of train data over all samples
        fxtest_samples = np.ones((samples, testsize))  # fx of test data over all samples

	# STORE RMSE FOR EVERY STEP AS THE MCMC PROGRESSES
        rmse_train = np.zeros(samples)
        rmse_test = np.zeros(samples)

	# WE INITIALISE THE WEIGHTS RANDOMLY  
        w = np.random.randn(w_size)
        w_proposal = np.random.randn(w_size)

	# THESE PARAMETERS CONTROL THE RANDOM WORK
	# THE FIRST THE CHANGES TO THE NETWORK WEIGHTS
        step_w = 0.02;  
	# THE SECOND THE VARIATION IN THE NOISE DISTRIBUTION
        step_eta = 0.01;

	#-------------------------------------------------------------------------------
        # --------------------- Declare FNN and initialize

        neuralnet = Network(self.topology, self.traindata, self.testdata)
        print('evaluate Initial w')

	# PASS THE DATA THROUGH THE NETWORK AND GET THE OUTPUTS ON BOTH TRAIN AND TEST
        pred_train = neuralnet.evaluate_proposal(self.traindata, w)
        pred_test = neuralnet.evaluate_proposal(self.testdata, w)

	#
	# INITIAL VALUE OF TAU IS BASED ON THE ERROR OF THE INITIAL NETWORK ON TRAINING DATA
	# ETA - IS USED FOR DOING THE RANDOM WALK SO THAT WE CAN ADD OR SUBTRACT RANDOM VALUES
	#       SUPPORT OVER [-INF, INF]
	# EXPONENTIATED TO GET tau_squared of the proposal WITH SUPPORT OVER [0, INF] 
        eta = np.log(np.var(pred_train - y_train))
        tau_pro = np.exp(eta)

	# 
	# THESE VALUES CONTROL THE INVERSE GAMMA FUNCTION
	# WHICH IS WHAT WE ASSUME TAU SQUARED IS DRAWN FROM
	# THIS IS CHOSEN FOR PROPOERTIES THAT COMPLIMENT WITH THE GAUSSIAN LIKELIHOOD FUNCTION
	# IS THERE A REFERENCE FOR THIS?
        nu_1 = 0
        nu_2 = 0

	# SIGMA SQUARED IS THE ASSUMED VARIANCE OF THE PRIOR DISTRIBUTION 
	# OF ALL WEIGHTS AND BIASES IN THE NEURAL NETWORK
        sigma_squared = 25

	#----------------------------------------------------------------------------------------
	# THIS NEXT SECTION INVOLVES CALCULATING: Metropolis-Hastings Acceptance Probability
	# This is what will determine whether a given change to the model weights (a proposal) 
	# is accepted or rejected
	# This will consist of the following
	# 1) A ratio of the likelihoods (current and proposal)
	# 2) A ratio of the priors (current and proposal)
	# 3) The inverse ratio of the transition probabilities. 
	# (Ommitted in this case because transitions are symetric)
	#----------------------------------------------------------------------------------------

	# PRIOR PROBABILITY OF THE WEIGHTING SCHEME W
        prior_current = self.log_prior(sigma_squared, nu_1, nu_2, w, tau_pro)

        # LIKELIHOOD OF THE TRAINING DATA GIVEN THE PARAMETERS
        [likelihood, pred_train, rmsetrain] = self.log_likelihood(neuralnet, self.traindata, w, tau_pro)

        # CALCULATED ON THE TEST SET FOR LOGGING AND OBSERVATION
        [likelihood_ignore, pred_test, rmsetest] = self.log_likelihood(neuralnet, self.testdata, w, tau_pro)

        print('Likelihood: ', likelihood)

        naccept = 0
        print('begin sampling using mcmc random walk')
        plt.plot(x_train, y_train)
        plt.plot(x_train, pred_train)
        plt.title("Plot of Data vs Initial Fx")
        plt.savefig('mcmcresults/begin.png')
        plt.clf()

        plt.plot(x_train, y_train)

        for i in range(samples - 1):

            w_proposal = w + np.random.normal(0, step_w, w_size)

            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)

            incl_pro = self.modifyIncludedData(incl)
            dataThisRound = self.reduceData(self.traindata, incl_pro)

            # WE RUN THIS TWICE SO WE HAVE PERFORMANCE OF THE WHOLE TRAINING SET FOR LOGGING
            [likelihood_proposal, preds, rmse] = self.log_likelihood(neuralnet, dataThisRound, w_proposal, tau_pro)
            [likelihood_train, pred_train, rmsetrain] = self.log_likelihood(neuralnet, self.traindata, w_proposal, tau_pro)
            [l_ignore, pred_test, rmsetest] = self.log_likelihood(neuralnet, self.testdata, w_proposal, tau_pro)

            # l_ignore  refers to parameter that will not be used in the alg.
            prior_prop = self.log_prior(sigma_squared, nu_1, nu_2, w_proposal, tau_pro)

            diff_likelihood = likelihood_proposal - likelihood
            diff_prior = prior_prop - prior_current

            mh_prob = min(1, math.exp(diff_likelihood + diff_prior))

            u = random.uniform(0, 1)

            if u < mh_prob:
                # Update position
                print ( i, ' is accepted sample')
                print ( round(incl.mean()*100,1), '% of data included' ) 
                naccept += 1
                likelihood = likelihood_proposal
                prior_current = prior_prop
                w = w_proposal
                eta = eta_pro
                incl = incl_pro

                print ( likelihood, prior_current, rmsetrain, rmsetest, w, 'accepted')

                pos_w[i + 1,] = w_proposal
                pos_tau[i + 1,] = tau_pro
                pos_incl[i + 1,] = incl_pro
                fxtrain_samples[i + 1,] = pred_train
                fxtest_samples[i + 1,] = pred_test
                rmse_train[i + 1,] = rmsetrain
                rmse_test[i + 1,] = rmsetest

                plt.plot(x_train, pred_train)


            else:
                pos_w[i + 1,] = pos_w[i,]
                pos_tau[i + 1,] = pos_tau[i,]
                pos_incl[i + 1,] = pos_incl[i,]
                fxtrain_samples[i + 1,] = fxtrain_samples[i,]
                fxtest_samples[i + 1,] = fxtest_samples[i,]
                rmse_train[i + 1,] = rmse_train[i,]
                rmse_test[i + 1,] = rmse_test[i,]

                # print ( i, 'rejected and retained')

        print (naccept, ' num accepted')
        print (naccept / (samples * 1.0), '% was accepted')
        accept_ratio = naccept / (samples * 1.0) * 100

        plt.title("Plot of Accepted Proposals")
        plt.savefig('mcmcresults/proposals.png')
        plt.savefig('mcmcresults/proposals.svg', format='svg', dpi=600)
        plt.clf()


        # Marginal posterior distribution of data inclusion
        pos_incl_marginal = pos_incl.mean(axis=0) 
        bars = plt.bar(list(range(0, len(pos_incl_marginal))), pos_incl_marginal)
        plt.title("Plot of Accepted Proposals")
        plt.savefig('mcmcresults/included_data_posterior.png')
        plt.savefig('mcmcresults/included_data_posterior.svg', format='svg', dpi=600)
        plt.clf()

        return (pos_w, pos_tau, fxtrain_samples, fxtest_samples, x_train, x_test, rmse_train, rmse_test, accept_ratio)


def main():
    outres = open('mcmcresults/resultspriors.txt', 'w')
    for problem in range(2, 3): 

        hidden = 5
        input = 4
        output = 1

        if problem == 1:
            traindata = np.loadtxt("Data_OneStepAhead/Lazer/train.txt")
            testdata = np.loadtxt("Data_OneStepAhead/Lazer/test.txt") 
        if problem == 2:
            traindata = np.loadtxt("Data_OneStepAhead/Sunspot/train.txt")
            testdata = np.loadtxt("Data_OneStepAhead/Sunspot/test.txt") 
        if problem == 3:
            traindata = np.loadtxt("Data_OneStepAhead/Mackey/train.txt")
            testdata = np.loadtxt("Data_OneStepAhead/Mackey/test.txt") 

        print(traindata)

        topology = [input, hidden, output]

	# Stop when RMSE reaches MinCriteria (Problem dependent)
        MinCriteria = 0.005  

        random.seed( time.time() )

	# Need to decide yourself
        numSamples = 80000  

        mcmc = MCMC(numSamples, traindata, testdata, topology)  # declare class

        [pos_w, pos_tau, fx_train, fx_test, x_train, x_test, rmse_train, rmse_test, accept_ratio] = mcmc.sampler()
        print('Sucessfully Sampled')

        burnin = 0.1 * numSamples  # Use post burn in samples

        pos_w = pos_w[int(burnin):, ]
        pos_tau = pos_tau[int(burnin):, ]

        fx_mu = fx_test.mean(axis=0)
        fx_high = np.percentile(fx_test, 95, axis=0)
        fx_low = np.percentile(fx_test, 5, axis=0)

        fx_mu_tr = fx_train.mean(axis=0)
        fx_high_tr = np.percentile(fx_train, 95, axis=0)
        fx_low_tr = np.percentile(fx_train, 5, axis=0)

        rmse_tr = np.mean(rmse_train[int(burnin):])
        rmsetr_std = np.std(rmse_train[int(burnin):])
        rmse_tes = np.mean(rmse_test[int(burnin):])
        rmsetest_std = np.std(rmse_test[int(burnin):])
        print (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std)
        np.savetxt(outres, (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, accept_ratio), fmt='%1.5f')

        ytestdata = testdata[:, input]
        ytraindata = traindata[:, input]

        plt.plot(x_test, ytestdata, label='actual')
        plt.plot(x_test, fx_mu, label='pred. (mean)')
        plt.plot(x_test, fx_low, label='pred.(5th percen.)')
        plt.plot(x_test, fx_high, label='pred.(95th percen.)')
        plt.fill_between(x_test, fx_low, fx_high, facecolor='g', alpha=0.4)
        plt.legend(loc='upper right')

        plt.title("Plot of Test Data vs MCMC Uncertainty ")
        plt.savefig('mcmcresults/mcmcrestest.png')
        plt.savefig('mcmcresults/mcmcrestest.svg', format='svg', dpi=600)
        plt.clf()
        # -----------------------------------------
        plt.plot(x_train, ytraindata, label='actual')
        plt.plot(x_train, fx_mu_tr, label='pred. (mean)')
        plt.plot(x_train, fx_low_tr, label='pred.(5th percen.)')
        plt.plot(x_train, fx_high_tr, label='pred.(95th percen.)')
        plt.fill_between(x_train, fx_low_tr, fx_high_tr, facecolor='g', alpha=0.4)
        plt.legend(loc='upper right')

        plt.title("Plot of Train Data vs MCMC Uncertainty ")
        plt.savefig('mcmcresults/mcmcrestrain.png')
        plt.savefig('mcmcresults/mcmcrestrain.svg', format='svg', dpi=600)
        plt.clf()

        mpl_fig = plt.figure()
        ax = mpl_fig.add_subplot(111)

        ax.boxplot(pos_w)

        ax.set_xlabel('[W1] [B1] [W2] [B2]')
        ax.set_ylabel('Posterior')

        #plt.legend(loc='upper right')

        plt.title("Boxplot of Posterior W (weights and biases)")
        plt.savefig('mcmcresults/w_pos.png')
        plt.savefig('mcmcresults/w_pos.svg', format='svg', dpi=600)

        plt.clf()


if __name__ == "__main__": main()
