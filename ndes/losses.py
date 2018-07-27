import numpy as np
import theano
import theano.tensor as T

def log_sum_exp(x, axis=None):
    x_max = T.max(x, axis=axis, keepdims=True)
    return T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max

def mapping(true, parameters):
        
    NT = T.shape(true)[0]
    D = T.shape(true)[1]
    M = T.shape(parameters)[1] // (D + D**2 + 1)
        
    means = parameters[:, : D*M].reshape((NT, M, D))
    sigmas = parameters[:, D*M : D*M + M*D*D].reshape((NT, M, D, D))
    weights = T.nnet.softmax(parameters[:, D*M + M*D*D:])
    return means, sigmas, weights

def neg_log_normal_mixture_likelihood(true, parameters):
    
    NT = T.shape(true)[0]
    D = T.shape(true)[1]
    M = T.shape(parameters)[1] // (D + D**2 + 1)
    means, sigmas, weights = mapping(true, parameters)
    two_pi = 2*np.pi
    log2pi = np.log(two_pi)
        
    def log_single_data_point(i, means, sigmas, weights, true):
        mu = means[i, :, :]
        P = sigmas[i, :, :, :]
        al = weights[i, :]
        tr = true[i,:]
            
        def log_single_component(c, mu, P, al, tr):
            L = T.tril(P[c,:,:], k=-1) + T.diag(T.exp(T.diagonal(P[c,:,:])))
            z = T.exp(-0.5*T.sum(T.dot(T.transpose(L), (tr - mu[c,:]))**2) + T.log(al[c]) + T.log(T.nlinalg.det(L)) - D*log2pi/2.)
            return z
            
        z, _ = theano.scan(fn=log_single_component,
                               sequences=T.arange(M),
                               non_sequences=[mu, P, al, tr])
        return T.log(T.sum(z)+1e-44)
        
    Z, _ = theano.scan(fn=log_single_data_point,
                           sequences=T.arange(NT),
                           non_sequences=[means, sigmas, weights, true])
    return -T.mean(Z)
