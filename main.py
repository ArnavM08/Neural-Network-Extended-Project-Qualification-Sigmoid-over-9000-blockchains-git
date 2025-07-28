import numpy as numpy
import matplotlib.pyplot as plt
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

def approximationCubicFunctionMSE(observed,epoch):
    assert str(type(observed)) == "<class 'list'>"
    m=len(observed)
    #feature normalization
    # 1,x,x^2,x^3:0,1,2,3
    sum1=0
    sum2=0
    sum3=0
    for i in range(m):
        x=float(observed[i][0])
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
        x=float(observed[i][0])
        sum1 = sum1 + ((x - mew1)**2) 
        sum2 = sum2 + (((x)**2 - mew2)**2) 
        sum3 = sum3 + (((x)**3 - mew3)**2)
    sigma1 = numpy.sqrt((1/(m-1)) * sum1)
    sigma2 = numpy.sqrt((1/(m-1)) * sum2)
    sigma3 = numpy.sqrt((1/(m-1)) * sum3)
    om1=[]
    om2=[]
    om3=[]
    for i in range(m):
        x=float(observed[i][0])
        om1.append(((x)-mew1)/sigma1)
        om2.append(((x**2)-mew2)/sigma2)
        om3.append(((x**3)-mew3)/sigma3)
    count=1
    a=0
    b=0
    c=0
    d=0
    sum1=0
    sum2=0
    sum3=0
    sum4=0
    errorList=[]
    #all good up to this mess
    while epoch > count:
        sum1=0
        sum2=0
        sum3=0
        sum4=0   
        sum5=0
        alpha=0.01     
        for i in range(m):
            predicted = a*om3[i] + b*om2[i] + c*om1[i] + d
            error=(predicted - observed[i][1])
            sum1 = sum1 + error
            sum2 = sum2 + error * om1[i]
            sum3 = sum3 + error * om2[i]
            sum4 = sum4 + error * om3[i]
            sum5= sum5 + abs(error)
        errorList.append(float(sum5))
        PDd = 1/m * sum1
        PDc = 1/m * sum2 
        PDb = 1/m * sum3
        PDa = 1/m * sum4
        a=a-(alpha * PDa)
        b=b-(alpha * PDb)
        c=c-(alpha * PDc)
        d=d-(alpha * PDd)
        count=count+1
    sum6=0
    for i in range(m):
        predicted = a*om3[i] + b*om2[i] + c*om1[i] + d
        error = (predicted - observed[i][1])        
        sum6 = sum6 + abs(error)
    errorList.append(float(sum6))
    return [a,b,c,d,mew1,mew2,mew3,sigma1,sigma2,sigma3,errorList]
array = [[0,1],[0.5,-4.875],[2.5,-43.375],[5,-69]] 
parameters=approximationCubicFunctionMSE(array,10000)
errorList=parameters[10]
print("model parameters for NORMALIZED features: f(x) = ",str(parameters[0]),"x_norm^3 + ",parameters[1],"x_norm^2 + ",parameters[2],"x_norm + ",parameters[3])
sum=0
for i in range(len(array)):
    x = float(array[i][0])
    normalizedX = (x - parameters[4])/parameters[7]
    normalizedX2 = (x**2 - parameters[5])/parameters[8]
    normalizedX3 = (x**3 - parameters[6])/parameters[9]
    prediction = (parameters[0] * normalizedX3) + (parameters[1] * normalizedX2) + (parameters[2] * normalizedX) + parameters[3]
    sum = sum + abs(prediction - array[i][1])
print("the model was off by ",sum," units")
plt.plot(errorList)
plt.xlabel('Epochs')
plt.ylabel('Error Value')
plt.show()
"""
cool looking datasets
[71,53],[58,111],[114,25] smooth gradient 
[0,1],[0.5,-4.875],[2.5,-43.375],[5,-69] local mimima
"""