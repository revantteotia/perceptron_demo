import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings(action='ignore') # to ignore matplotlib warnings


def plotPointsWithhyperplane(trainingX, trainingY, W, saveFileName="perceptron.png"):

    '''
    To plot training points along with the hyperplane defined by weights W
    '''

    plt.figure(figsize=(10,10)) # to get a new figure each time
    plt.grid(True, alpha=0.5)

    # Plotting the points
    color= ['red' if y == 1 else 'green' for y in trainingY]
    plt.scatter(trainingX[:,0], trainingX[:,1], color=color)
    plt.ylim(-1.5, 1.5)
    plt.xlim(-1.5, 1.5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    
    # plotting W vector : use normalized W vector to show on plot
    W_normalized = W / np.sqrt( np.dot(W,W) )
    plt.arrow(0, 0, W_normalized[0]/5, W_normalized[1]/5, head_width=0.05, head_length=0.05, fc='k', ec='k', label='W')
    plt.annotate("W", xy=(W_normalized[0]/5, W_normalized[1]/5), xytext=(0, 10), textcoords="offset points")


    # plotting hyperplane : note that hyperplane is perpendicular to the W vector
    x1 = np.linspace(-2,2,num=5)
    if (W[1] != 0):
        x2 = (-W[0]/W[1])*x1
        
    else: # printing a vertical/no hyperplane
        if (W[0] == 0):
            x1 = 0*x1
            x2 = x1
        else:
            x2 = x1
            x1 = 0*x1    
        
    hyperplane, = plt.plot(x1, x2, label='Hyperplane')

    # shading the regions
    plt.fill_between([0,1.5], 1.5, -1.5, where=(W[1] == 0 and W[0] > 0 ), color='red', alpha=0.3)
    plt.fill_between([-1.5,0], 1.5, -1.5, where=(W[1] == 0 and W[0] > 0 ), color='green', alpha=0.3)
    plt.fill_between([0,1.5], 1.5, -1.5, where=(W[1] == 0 and W[0] < 0 ), color='green', alpha=0.3)
    plt.fill_between([-1.5,0], 1.5, -1.5, where=(W[1] == 0 and W[0] < 0 ), color='red', alpha=0.3)
    plt.fill_between(x1, x2, -1.5, where=(W[1] < 0), color='red', alpha=0.3)
    plt.fill_between(x1, 1.5, x2, where=(W[1] > 0), color='red', alpha=0.3)
    plt.fill_between(x1, x2, -1.5, where=(W[1] > 0), color='green', alpha=0.3)
    plt.fill_between(x1, 1.5, x2, where=(W[1] < 0), color='green', alpha=0.3)

    # adding legend
    red_patch = mpatches.Patch(color='red', label='+1 class')
    green_patch = mpatches.Patch(color='green', label='-1 class')
    
    plt.legend(loc="upper left", handles=[red_patch, green_patch, hyperplane])
    
    # Setting Weights as title
    plt.title("W=[{:.2f}, {:.2f}]".format(W[0],W[1]))

    plt.tight_layout()
    plt.savefig(saveFileName)  


def trainPerceptron(trainingX, trainingY, initialWeights, learningRate=1):
    '''
    Trains a perceptron on given training data. Plots the training points and learned hyperplane after every update


    Arguments :
        trainingX : 2D training data        
        trainingX : labels, can be -1 or +1
        ititialWeights : weights initialized
        learningRate : to update weights

    Returns : 
        updatesNeeded : number of updates needed to converge
        W : final weights

    '''

    

    MAX_EPOCHS = 50 # to stop training in case the perceptron does not converge 
    learningRate = learningRate
    W = initialWeights
    print("Initial weights = {}".format(W))


    # Plotting the hyperplane with initital weights
    plotPointsWithhyperplane(trainingX, trainingY, W, saveFileName= "initial_weights.png")


    updatesNeeded = 0
    for _ in range(MAX_EPOCHS):
        updatedWeightsInThisEpoch = False

        for point, label in zip(trainingX, trainingY):
            isMistakePoint = False
            WUpdate = np.zeros_like(W)

            if (label == 1 and np.dot(W,point) < 0): # true class = 1, predicted class = -1
                isMistakePoint = True
                WUpdate = learningRate * point
                
            
            elif(label == -1 and np.dot(W,point) >= 0): # true class = -1, predicted class = 1
                isMistakePoint = True
                WUpdate = (-1) * learningRate * point
                

            if (isMistakePoint):     # to log/plot update
                print("===================")
                print("mistaken point = {}".format(point))
                print("old weights = {}".format(W))
                W = W + WUpdate
                print("change in Weights = {}".format(WUpdate))
                print("New weights = {}".format(W))
                print("===================")
                updatesNeeded += 1
                updatedWeightsInThisEpoch = True
                plotPointsWithhyperplane(trainingX, trainingY, W, saveFileName= "after_{}_updates.png".format(updatesNeeded))
        
        if (updatedWeightsInThisEpoch == False): # i.e. no mistakes --> model has converged 
            break

    return updatesNeeded, W
        

if  __name__ == "__main__":

    trainingX = np.array([ [1,1], [-1,-1], [0,0.5], [0.1,0.5], [0.2,0.2], [0.9,0.5] ], dtype=float)
    trainingY = np.array( [1, -1, -1, -1, 1, 1], dtype=float )
    W = np.array( [1,1], dtype=float )

    updatesNeeded, finalWeights = trainPerceptron(trainingX, trainingY, W)
    print("{} number of updates needed to converge".format(updatesNeeded))
    print("Final Weights : {}".format(finalWeights))
