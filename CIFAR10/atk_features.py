import numpy as np
from numpy.lib.recfunctions import append_fields
from matplotlib import pyplot


# Receive data array and return structured array
def structArray(X):
    # Define the datatype for the initial structured array
    dt = {'names':['p'+ str(i) for i in range(len(X[0]))], 'formats':['float64' for i in range(len(X[0]))]}
    # Create the structured array
    X_structured = np.zeros(shape=X.shape[0], dtype=dt)
    # Populate the structured array
    for c in range(X.shape[1]):
        X_structured['p'+str(c)] = X[:,c]
    return X_structured


# Add feature fields to structured array
def ewFeat(X):

    # Variance
    X = append_fields(X, names='evar', data=[np.var(list(i)) for i in X], usemask=False)

    # Other features?

    return X


# Plot analysis artifacts related to data set
def analyze(filename, X, Y):

    # Inspect each feature
    for feat in X.dtype.names:
        # Plot histograms of each feature
        nbins= 50
        pyplot.hist(X[Y==0.0][feat], nbins, alpha=0.5, label='Benign')
        pyplot.hist(X[Y==1.0][feat], nbins, alpha=0.5, label='Malicious')
        pyplot.legend(loc='upper right')
        pyplot.title(filename+'_' + feat)
        pyplot.xlabel(feat)
        pyplot.ylabel('frequency')
        pyplot.savefig('./analysis/'+filename+'_'+feat+'.png')
        pyplot.close()

        # Other stats/plots?

    return





