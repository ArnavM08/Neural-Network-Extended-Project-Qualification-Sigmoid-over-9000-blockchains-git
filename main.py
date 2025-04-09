#dataset an array of coordinates, so [(0,2),(3,5)] predicted follows a similar format, but takes x coordinate from dataset and uses model to find predicted y val
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