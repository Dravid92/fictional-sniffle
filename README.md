# Project 3 - Predicting Insurance amount and Smokers (Linear Regression and KNN) 

## Introduction :
  
    Machine Learning with R by Brett Lantz is a book that provides an introduction to machine learning using R. 
    As far as I can tell, Packt Publishing does not make its datasets available online unless you buy the book and 
    create a user account which can be a problem if you are checking the book out from the library or borrowing the
    book from a friend. All of these datasets are in the public domain but simply needed some cleaning up and recoding
    to match the format in the book.

## Data reference : 

## https://www.kaggle.com/mirichoi0218/insurance

## Task 1 - Predict the Insurance Cost 

## Task 2 - Predict Whether the person Smokes 

## Variables :
- **age**: age of primary beneficiary

- **sex**: insurance contractor gender, female, male

- **bmi**: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9

- **children**: Number of children covered by health insurance / Number of dependents

- **smoker**: Smoking

- **region**: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.

- **charges**: Individual medical costs billed by health insurance

## Pre- Processing :
### Recoding:

- Variables such as **gender, smoker ,bmi and Region** are considered as ordinal factors and are recoded.

![Capture](https://user-images.githubusercontent.com/41041795/94027226-40ca8980-fdd8-11ea-8b31-f75ee27028e7.PNG)
- BMI is recoded as the following   
***BMI** > 25*  
*18 < **BMI** <25*  
***BMI** < 25*
 
![2](https://user-images.githubusercontent.com/41041795/94027651-b2a2d300-fdd8-11ea-90d7-dd2255225098.PNG)
## Task 1 - Predict the Insurance Cost 
### Model 
 - Linear Regression is used to predict Insurance cost 
 
 ![3](https://user-images.githubusercontent.com/41041795/94028266-61471380-fdd9-11ea-908a-2d96f2fbed20.PNG)

### Prediction 

- Inputs are age , sex and bmi to predict the cost of insurance for a person 

![4](https://user-images.githubusercontent.com/41041795/94028985-f9dd9380-fdd9-11ea-8534-2a9be3c7c71b.PNG)


## Task 2 - Predict Whether the person Smokes 
### Model 
 - Using K- Nearest Neighbors smokers where predicted 
 
```python
# Using k-Nearest neighbours to predict a smoker
def smokers_():
    # X and y data
    X = np.array(data.drop(['smoker','children','region'],1))
    y = np.array(data['smoker'])
    # char is the insurance amount
    char = input('\nWhat was the cost of insurance ! ')

    x = [[age,sex,bmi,char]]
    # passing the x values as float
    pred = np.array(x,float)
    # spliting the data for testing and training
    X_train , X_test,y_train , y_test = cross_validation.train_test_split(X,y,test_size=0.2)
    # the KNN function
    clf = neighbors.KNeighborsClassifier()
    # Training
    clf.fit(X_train,y_train)
    # Testing
    acc = clf.score(X_test,y_test)
    #prediction
    predictor = clf.predict(pred)
    if predictor == 0:
        print('with 90 % accuracy I can say that the person is not a Smoker !\n')
    else:
        print('Unfortunately the data says that he is a smoker ! \n')
    print( str(acc) + 'is the accuracy for predicting a smoker  \n')
smokers_()
```
