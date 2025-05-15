import numpy as numpy
#dataset an array of coordinates, so [(0,2),(3,5)] predicted follows a similar format, but takes x coordinate from dataset and uses model to find predicted y val
#LOSS FUNCTIONS
def MSE(dataset,predicted):
    assert len(dataset) == len(predicted)
    assert type(dataset) == "list"
    assert type(predicted) == "list"
    total=0
    for i in range(len(dataset)):
        total = total + ((dataset[i][1] - predicted[i][2])**2) 
    return total * 1/len(dataset)
def MAE(dataset,predicted):
    assert len(dataset) == len(predicted)
    assert type(dataset) == "list"
    assert type(predicted) == "list"
    total=0
    for i in range(len(dataset)):
        total= total + abs(dataset[i][1] - predicted[i][1])
    return total * 1/len(dataset)
def Huber(dataset,predicted,thresholdValue):
    assert len(dataset) == len(predicted)
    assert type(dataset) == "list"
    assert type(predicted) == "list"
    assert type(thresholdValue) == "float"
    total=0
    for i in range(len(dataset)):
        if abs(dataset[i][1] - predicted[i][1]) <= thresholdValue:
            total=total+ (1/2*((dataset[i][1] - predicted[i][1])**2))
        else:
            total=total + (thresholdValue * abs(dataset[i][1] - predicted[i][1]) - ((1/2)*(thresholdValue**2)))
    return total * 1/len(dataset)
#before constructing activation functions, need to have a node to operate on, and for that i may as well just create the whole network data structure
#idea is that a doubly linked list works well for my needs, with a global array of parameters that are used for the transforms from one node to the rest
#for the meantime, ill handle this in the main
#ACTIVATION FUNCTIONS
#kinda implied i use non-linear activation functions

def sigmoid(weightedValue):
    assert type(weightedValue) == "float"
    return 1/(1+(numpy.e**weightedValue))
def tanh(weightedValue):
    assert type(weightedValue) == "float"
    return ((numpy.e**weightedValue) - (numpy.e**(-weightedValue)))/((numpy.e**weightedValue)+(numpy.e**(-weightedValue)))
#this is somehow not a linear function
def RELU(weightedValue):
    assert type(weightedValue) == "float"
    return max(weightedValue,0)
def leakyRELU(weightedValue):
    assert type(weightedValue) == "float"
    return max(0.1*weightedValue,weightedValue)
def paramRELU(weightedValue,constant):
    assert type(weightedValue) == "float"
    assert type(constant) == "float"
    return max((constant*weightedValue),weightedValue)
def ELU(weightedValue,constant):
    assert type(weightedValue) == "float"
    assert type(constant) == "float"
    if weightedValue < 0:
        return (constant*((numpy.e**weightedValue) - 1))
    elif weightedValue >= 0:
        return weightedValue
def swish(weightedValue):
    assert type(weightedValue) == "float"
    return weightedValue/(1+(numpy.e**(-weightedValue)))
def GELU(weightedValue):
    assert type(weightedValue) == "float"
    #arnavs gonna start frothing at the mouth (i did not)
    return 0.5*weightedValue(1+ (((numpy.e**((numpy.sqrt((2/numpy.pi))*weightedValue * 0.044715*(weightedValue ** 3)))-numpy.e**(numpy.sqrt((2/numpy.pi))*weightedValue * 0.044715*(weightedValue ** 3)))/(numpy.e**(numpy.sqrt((2/numpy.pi))*weightedValue * 0.044715*(weightedValue ** 3))+numpy.e**(numpy.sqrt((2/numpy.pi))*weightedValue * 0.044715*(weightedValue ** 3))))))
def SELU(weightedValue,constant1,constant2):
    assert type(weightedValue) == "float"
    assert type(constant1) == "float"
    assert type(constant2) == "float"
    if weightedValue < 0: 
        return constant2 * (constant1*((numpy.e** weightedValue)-1))
    else:
        return constant2 * weightedValue