import numpy as np # linear algebra
from sklearn.preprocessing import label_binarize


class GaussianNB():
    """
    Gaussian Naive Bayes 

    Assume each class conditional feature distribution is
    independent and estimate the mean and variance from the
    training data

    Parameters
    ----------
    epsilon: float
        a value that add to variance to prevent numerical error
    
    Attributes
    ----------
    num_class : ndarray of shape (n_classes,)
        count of each class in the training sample

    mean: ndarray of shape (n_classes,n_features)
            mean of each variance
    
    variance: ndarray of shape (n_classes,n_features)
        variance of each variance
    
    prior :  ndarray of shape (n_classes,)
            probability of each class

    """
    def __init__(self,eps=1e-6):
        self.eps = eps 
    
    def fit(self,X,y):
        """
        Train the model with X,y

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Input data
        y: ndarray of shape (n_samples,)
            Target
        
        returns
        --------
        self: object
        """
        
        self.n_sample, self.n_features = X.shape
        self.labels = np.unique(y)
        self.n_classes = len(self.labels)

        self.mean = np.zeros((self.n_classes,self.n_features))
        self.sigma = np.zeros((self.n_classes,self.n_features))
        self.prior = np.zeros((self.n_classes,))

        for i in range(self.n_classes):
            X_c = X[y==i,:]

            self.mean[i,:] = np.mean(X_c,axis=0)
            self.sigma[i,:] = np.var(X_c,axis=0) + self.eps
            self.prior[i] = X_c.shape[0]/self.n_sample

        return self

    def predict(self,X):
        if X.shape[1] != self.n_features:
            err = 'number of features should match the training data'
            raise ValueError(err)

        probs = np.zeros((X.shape[0],self.n_classes))
        for i in range(self.n_classes):
             probs[:,i] = self.prob(X,self.mean[i,:],self.sigma[i,:],self.prior[i])


        return probs

    def prob(self,X,mean,sigma,prior):

        prob = -self.n_features / 2 * np.log(2 * np.pi) - 0.5 * np.sum(
            np.log(sigma + self.eps)
            )
        prob -= 0.5 * np.sum(np.power(X -mean, 2) / (sigma + self.eps), 1)

        return prior + prob


class multinomialNB():
    def __init__(self,alpha=1.0):
        self.alpha = alpha

    def fit(self,X,y):
        self.n_samples , self.n_features = X.shape
        self.classes = np.unique(y)
        self.y = label_binarize(y,classes=self.classes)
        if self.y.shape[1] == 1:
                self.y = np.concatenate((1 -  self.y,  self.y), axis=1)


        self.feature_count_      =  self.y.T @ X + self.alpha
        self.word_count_byClass  = np.sum( self.feature_count_,axis=1).reshape(-1,1) + self.alpha*self.n_features

        self.prior =  self.feature_count_ /self.word_count_byClass

    def predict(self,x):
        prob = x @ self.prior.T
        return  prob


