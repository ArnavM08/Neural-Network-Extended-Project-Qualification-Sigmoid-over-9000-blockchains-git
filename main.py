import numpy as numpy
import random as random
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
    assert str(type(weightedValue)) == "float"
    return weightedValue/(1+(numpy.e**(-weightedValue)))
def GELU(weightedValue):
    assert str(type(weightedValue)) == "float"
    #arnavs gonna start frothing at the mouth
    return 0.5*weightedValue(1+ (((numpy.e**((numpy.sqrt((2/numpy.pi))*weightedValue * 0.044715*(weightedValue ** 3)))-numpy.e**(numpy.sqrt((2/numpy.pi))*weightedValue * 0.044715*(weightedValue ** 3)))/(numpy.e**(numpy.sqrt((2/numpy.pi))*weightedValue * 0.044715*(weightedValue ** 3))+numpy.e**(numpy.sqrt((2/numpy.pi))*weightedValue * 0.044715*(weightedValue ** 3))))))
def SELU(weightedValue,constant1,constant2):
    assert str(type(weightedValue)) == "float"
    assert str(type(constant1)) == "float"
    assert str(type(constant2)) == "float"
    if weightedValue < 0: 
        return constant2 * (constant1*((numpy.e** weightedValue)-1))
    else:
        return constant2 * weightedValue
#let our approximation function be able to model up to a cubic graph. this means our function will look like f(x,a,b,c,d) = ax^3 + bx^2 + cx + d
#given our partial derivatives for each variable excluding x, we can adjust and optimize each of our variables until our error functions either return a model with a feasibly low error value, or an exact match.
#this will be a stochastic algorithm, meaning the order in which our variables are adjusted is completely random.  
def approximationCubicFunctionMSE(observed):
    assert str(type(observed)) == "<class 'list'>"
    a=0
    b=0
    c=0
    d=0
    m=len(observed)
    #feature normalization
    # x,x^2,x^3:1,2,3
    sum1=0
    sum2=0
    sum3=0
    for i in range(m):
        x=int(observed[i][0])
        sum1 = sum1 + x
        sum2 = sum2 + x**2
        sum3 = sum3 + x**3
    mew1 = (1/m) * sum1
    mew2 = (1/m) * sum2
    mew3 = (1/m) * sum3
    sum1=0
    sum2=0
    sum3=0
    for i in range(m):
        x=int(observed[i][0])
        sum1 = sum1 + (x - mew1)**2 
        sum2 = sum2 + (x - mew2)**2 
        sum3 = sum3 + (x - mew3)**2 
    sigma1 = numpy.sqrt((1/m) * sum1)
    sigma2 = numpy.sqrt((1/m) * sum2)
    sigma3 = numpy.sqrt((1/m) * sum3)
    om1=[]
    om2=[]
    om3=[]
    for i in range(m):
        x=int(observed[i][0])
        om1.append(((x)-mew1)/sigma1)
        om2.append(((x**2)-mew2)/sigma2)
        om3.append(((x**3)-mew3)/sigma3)
    