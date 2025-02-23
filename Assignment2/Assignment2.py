import json
import gzip
import math
import numpy as np
from collections import defaultdict
from sklearn import linear_model
import random
import statistics
from sklearn.metrics import mean_squared_error
from statistics import median

def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N
    
answers = {}

z = gzip.open("steam.json.gz")
dataset = []
for l in z:
    d = eval(l)
    dataset.append(d)    
z.close()

### Question 1

def feat1(d):
    # your implementation\
    f = len(d['text'])
    return [1,f]

def MSE(y, ypred):
    if len(y) != len(ypred):
        raise ValueError("Lists must have the same length.")
    squared_errors = [(yi - ypredi) ** 2 for yi, ypredi in zip(y, ypred)]
    return sum(squared_errors) / len(squared_errors)

def predict_rating_func1(datum, theta):
    r_val = theta[0]
    a = len(datum['text'])
    if(len(theta)>1):
        for i in range(1, len(theta)):
            r_val += theta[i] * (a**i)
    return r_val

X_1 = [feat1(datum) for datum in dataset]
Y_1 = [datum['hours'] for datum in dataset]

theta1,residuals,rank,s = np.linalg.lstsq(X_1,Y_1)

predicted_ratings1 = [predict_rating_func1(datum,theta1) for datum in dataset]
actual_ratings1 = [datum['hours'] for datum in dataset]

mse1 = MSE(actual_ratings1, predicted_ratings1)

answers['Q1'] = [float(theta1[1]), float(mse1)] # Remember to cast things to float rather than (e.g.) np.float64
assertFloatList(answers['Q1'], 2)

## Question 2

dataTrain = dataset[:int(len(dataset)*0.8)]
dataTest = dataset[int(len(dataset)*0.8):]

under = 0
over = 0

def underpredicts(y, ypred):   
    x = [1 for a,b in zip(y, ypred) if b < a ]
    return sum(x) 
    
def overpredicts(y, ypred):    
    x = [1 for a,b in zip(y, ypred) if b > a ]
    return sum(x)

X_train = [feat1(datum) for datum in dataTrain]
Y_train = [datum['hours'] for datum in dataTrain]

X_test = [feat1(datum) for datum in dataTest]
Y_test = [datum['hours'] for datum in dataTest]

model2 = linear_model.LinearRegression()
model2.fit(X_train, Y_train)
y_pred = model2.predict(X_test)

mse2 = MSE(Y_test, y_pred)

under = underpredicts(Y_test, y_pred)
over =  overpredicts(Y_test, y_pred)

answers['Q2'] = [mse2, under, over]
assertFloatList(answers['Q2'], 3)

theta0 = model2.intercept_  # This is the intercept (theta0)

### Question 3
#a
y2 = Y_train[:]
y2.sort()
perc90 = y2[int(len(y2)*0.9)] # 90th percentile

dataTrain_a = [d for d in dataTrain if d['hours'] < perc90]
X3a = [feat1(datum) for datum in dataTrain_a]
y3a = [datum['hours'] for datum in dataTrain_a]

mod3a = linear_model.LinearRegression(fit_intercept=False)
mod3a.fit(X3a,y3a)
pred3a = mod3a.predict(X_test)

mse3a = MSE(Y_test, pred3a)

under3a = underpredicts(Y_test, pred3a)
over3a =  overpredicts(Y_test, pred3a)

#b

y3b = [datum['hours_transformed'] for datum in dataTrain]
x3b = [feat1(datum) for datum in dataTrain]

y3b_test = [datum['hours_transformed'] for datum in dataTest]

mod3b = linear_model.LinearRegression(fit_intercept=False)
mod3b.fit(x3b,y3b)
pred3b = mod3b.predict(X_test)

under3b = underpredicts(y3b_test, pred3b)
over3b  = overpredicts(y3b_test, pred3b)

#c

median_length = median([len(d['text']) for d in dataTrain])
median_hours = median(Y_train)

theta1 = (median_hours - theta0) / median_length

pred3c = [theta0 + theta1*len(d['text']) for d in dataTest]

under3c = underpredicts(Y_test, pred3c)
over3c = overpredicts(Y_test, pred3c)

answers['Q3'] = [under3a, over3a, under3b, over3b, under3c, over3c]
assertFloatList(answers['Q3'], 6)

## Question 4

median_value = median(Y_train)
Y_4_train = [1 if y > median_value else 0 for y in Y_train]
Y_4_test = [1 if y > median_value else 0 for y in Y_test]

mod4 = linear_model.LogisticRegression(C = 1)
mod4.fit(X_train, Y_4_train)
predictions = mod4.predict(X_test) # Binary vector of predictions

def conf_Matrix(Y_Pred, Y):
    TP = len([a for a,b in zip(Y_Pred, Y) if a == True and b == True])
    TN = len([a for a,b in zip(Y_Pred, Y) if a == False and b == False])
    FP = len([a for a,b in zip(Y_Pred, Y) if a == True and b == False])
    FN = len([a for a,b in zip(Y_Pred, Y) if a == False and b == True])
    TPR = (TP/(TP+FN))
    TNR = (TN/(TN+ FP))
    BER = 1 - (TPR+TNR)/2
    return [TP, TN, FP, FN, BER]

values = conf_Matrix(predictions, Y_4_test)

answers['Q4'] = values
assertFloatList(answers['Q4'], 5)

## Question 5

under5 = underpredicts(Y_4_test, predictions)
over5 =  overpredicts(Y_4_test, predictions)

answers['Q5'] = [over5, under5]
assertFloatList(answers['Q5'], 2)

## Question 6

## a

dataTrain_2014 = [entry for entry in dataTrain if int(entry['date'][:4])<= 2014]
dataTest_2014 = [entry for entry in dataTest if int(entry['date'][:4])<= 2014]

Y_train_2014 = [datum['hours'] for datum in dataTrain_2014]
Y_test_2014 = [datum['hours'] for datum in dataTest_2014 ]

median_value6a = median(Y_train_2014 )


X2014 = [feat1(datum) for datum in dataTrain_2014]
y2014 = [1 if y > median_value6a else 0 for y in Y_train_2014]

X2014test = [feat1(datum) for datum in dataTest_2014 ]
y2014test = [1 if y > median_value6a else 0 for y in Y_test_2014]

mod6a = linear_model.LogisticRegression(C = 1)
mod6a.fit(X2014, y2014)
pred6a = mod6a.predict(X2014test) # Binary vector of predictions

values6a = conf_Matrix(pred6a, y2014test)
BER_A = values6a[4]

## b

dataTrain_2015 = [entry for entry in dataTrain if int(entry['date'][:4]) >= 2015]
dataTest_2015 = [entry for entry in dataTest if int(entry['date'][:4])>= 2015]

Y_train_2015 = [datum['hours'] for datum in dataTrain_2015]
Y_test_2015 = [datum['hours'] for datum in dataTest_2015 ]

median_value6b = median(Y_train_2015 )

X2015 = [feat1(datum) for datum in dataTrain_2015]
y2015 = [1 if y > median_value6b else 0 for y in Y_train_2015]

X2015test = [feat1(datum) for datum in dataTest_2015]
y2015test = [1 if y > median_value6b else 0 for y in Y_test_2015]

mod6b = linear_model.LogisticRegression(C = 1)
mod6b.fit(X2015, y2015)
pred6b = mod6b.predict(X2015test) # Binary vector of predictions

values6b = conf_Matrix(pred6b, y2015test)
BER_B = values6b[4]

## c
mod6c = linear_model.LogisticRegression(C = 1)
mod6c.fit(X2014, y2014)
pred6c = mod6c.predict(X2015test) # Binary vector of predictions
values6c = conf_Matrix(pred6c, y2015test)
BER_C = values6c[4]

##d
mod6d = linear_model.LogisticRegression(C = 1)
mod6d.fit(X2015, y2015)
pred6d = mod6c.predict(X2014test) # Binary vector of predictions
values6d = conf_Matrix(pred6d, y2014test)
BER_D = values6d[4]

answers['Q6'] = [BER_A, BER_B, BER_C, BER_D]
assertFloatList(answers['Q6'], 4)

### Question 7

usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
reviewsPerItem = defaultdict(list)
ratingDict = {} # To retrieve a rating for a specific user/item pair

for d in dataTrain:
    user,item = d['userID'], d['gameID']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    reviewsPerUser[user].append(d)
    reviewsPerItem[item].append(d)  
    
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


def most_similar(first_user_id, N):
    similarities = []
    first_user_items = itemsPerUser[first_user_id]
    
    for u2 in itemsPerUser:
        if u2 != first_user_id: 
            similarity = Jaccard(first_user_items, itemsPerUser[u2])
            similarities.append((similarity, u2))

    similarities.sort(reverse=True)
    return similarities[:N]

first_user_id = dataTrain[0]['userID']
top_10_similar_users = most_similar(first_user_id, 10)

first = top_10_similar_users[0][0]
tenth = top_10_similar_users[9][0]

answers['Q7'] = [first, tenth]
assertFloatList(answers['Q7'], 2)

### Question 8

hoursMean = sum([d['hours_transformed'] for d in dataTrain]) / len(dataTrain)

def predictRating1(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerItem[item]:
        u2 = d['userID']
        if u2 == user: continue
        ratings.append(d['hours_transformed'])
        similarities.append(Jaccard(itemsPerUser[user],itemsPerUser[u2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return sum(weightedRatings) / sum(similarities)
    else:
        return hoursMean
    
def predictRating2(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['gameID']
        if i2 == item: continue
        ratings.append(d['hours_transformed'])
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return sum(weightedRatings) / sum(similarities)
    else:
        return hoursMean
    
def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)

labels = [d['hours_transformed'] for d in dataTest]
simPredictions_1 = [predictRating1(d['userID'], d['gameID']) for d in dataTest]
MSEU = MSE(simPredictions_1, labels)

simPredictions_2 = [predictRating2(d['userID'], d['gameID']) for d in dataTest]
MSEI = MSE(simPredictions_2, labels)


answers['Q8'] = [MSEU, MSEI]
assertFloatList(answers['Q8'], 2)

### Question 9
import math

def predictRating3(user,item, date):
    ratings = []
    similarities = []
    exp_year = []
    for d in reviewsPerItem[item]:
        u2 = d['userID']
        if u2 == user: continue
        ratings.append(d['hours_transformed'])
        sub_y = math.e ** (-abs(date - int(d['date'][:4])))
        exp_year.append(sub_y)
        similarities.append(Jaccard(itemsPerUser[user],itemsPerUser[u2]))
        
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        weightedRatings_year = [(x*y) for x,y in zip(weightedRatings,exp_year)]
        similarities_year = [(x*y) for x,y in zip(similarities,exp_year)]
        return sum(weightedRatings_year) / sum(similarities_year)
    else:
        return hoursMean
    
simPredictions_3 = [predictRating3(d['userID'], d['gameID'], int(d['date'][:4])) for d in dataTest]
MSE9 = MSE(simPredictions_3, labels)

answers['Q9'] = MSE9
assertFloat(answers['Q9'])

if "float" in str(answers) or "int" in str(answers):
    print("it seems that some of your answers are not native python ints/floats;")
    print("the autograder will not be able to read your solution unless you convert them to ints/floats")

f = open("answers_midterm.txt", 'w')
f.write(str(answers) + '\n')
f.close()