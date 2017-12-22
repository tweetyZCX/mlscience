import pickle
import numpy as np
class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.Weaker=weak_classifier
        self.n=n_weakers_limit 
        pass

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        self.W=np.ones((np.shape(X)[0]))/np.shape(X)[0]    
        self.α=[]
        self.Q=0 
        for i in range(self.n):    
            clf=self.Weaker.fit(X,y,self.W)
            AdaBoostClassifier.save(clf, "weak_classifier_%d.pkl"%i)
            h=clf.predict(X)
            e=1-clf.score(X,y,self.W)
            if e>0.5:
                break
            self.α.append(1/2*np.log((1-e)/e))
            Z=self.W*np.exp(-self.α[i]*y*h)
            self.W=(Z/np.sum(Z))
            self.Q=i 
            H=AdaBoostClassifier.predict(self,X,0)
            if np.sum(H!=y)==0:
                print(i+1,"weak classifier is enough to make error=0")
                break
            else:
                print(i+1,"weak classifier AdaBoost's accuancy is",1-(np.sum(H!=y)/np.shape(X)[0]))
        return self


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        pass

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        sums=np.zeros((np.shape(X)[0]))
        for i in range(self.Q+1):
            clf=AdaBoostClassifier.load("weak_classifier_%d.pkl"%i)
            sums=sums+clf.predict(X)*self.α[i] 
        return np.sign(sums)
        

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
