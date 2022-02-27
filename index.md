TODO:
Create a new Github page with a presentation on the concepts of Multivariate Regression Analysis and Gradient Boosting. Include a presentation of Extreme Gradient Boosting (xgboost).

Apply the regression methods (including lowess and boosted lowess) to real data sets, such as "Cars" and "Boston Housing Data".  Record the cross-validated mean square errors and the mean absolute errors.

For each method and data set report the crossvalidated mean square error and determine which method is achieveng the better results.

In this paper you should also include theoretical considerations, examples of Python coding and plots. 

The final results should be clearly stated.

## Overview

For project three, I applied two regression techniques, Locally Weighted Linear (LOWESS) and Random Forest, on the Boston Housing Dataset. Both regression techniques were then boosted - applying Boosted LOWESS and Extreme Gradient Boosting (XGBoost). The Boston Housing Dataset is derived from information collected by the U.S. Census Service concerning homes in the area of Boston, Massachusetts. Although the dataset contains over fifteen columns, each a distinct characteristic of a home in Boston, I will be focusing on only four, applying multivariate regression analysis techniques. The input features will be 'rooms', 'crime', and 'distance' describing the average number of rooms per dwelling, per capita crime rate by town, and weighted distances to five Boston employment centres. The target, which will be predicted, is 'cmedv', the median value of owener-occupied homes (in thousands of dollars).

Snippet of Dataset:
<img width="1019" alt="Screen Shot 2022-02-27 at 4 28 58 PM" src="https://user-images.githubusercontent.com/71660299/155900623-33829067-50e4-4d7d-9f78-c22768ffe3e7.png">


## Locally Weighted Linear Regression (LOWESS)
Stemming from Linear Regression, LOWESS can be considered a non-parametric algorithm that must use all the dataset for estimation (Figueira, 2021). 


Prediction Equation for Locally Weighted Linear Regression: 

<p align = 'center'> <img width="306" alt="Screen Shot 2022-02-12 at 11 19 23 PM" src="https://user-images.githubusercontent.com/71660299/153738433-1c60e39b-c996-48df-8b3d-4cea7e03a88f.png"> 
    
    - *yhat* is obtained as a different linear combination of the values of y



##### How did we get this LOWESS prediction equation from a simple linear equation?

First, linear regression - the assumption that: 
<p align = 'center'>    <img width="171" alt="Screen Shot 2022-02-12 at 11 15 56 PM" src="https://user-images.githubusercontent.com/71660299/153738371-656b4681-e83f-4daf-8322-3f9646b9b8d3.png">
    So, if we pre-multiply this equation with a **matrix** of weights we get: 
    <p align = 'center'> <img width="333" alt="Screen Shot 2022-02-12 at 11 19 48 PM" src="https://user-images.githubusercontent.com/71660299/153738443-5b0e3cc1-36f2-478e-a0f5-5302de978eef.png">. 
        Keep in mind here that *the "weights" are on the main diagonal and the rest of the elements are 0*. 
- The independent observations are the rows of the matrix *X* 
- Each row has a given number of columns (*number of features*), denoted by *p*. 
- Thus, every row is a vector in R^*p*. 
- The distance between two independent observations is the **Euclidean distance** between the two represented $p$-dimensional vectors. Euclidean distance is also commonly referred to as *L2 Norm*. 

As a result, this equation is as follows: 
  <p align = 'center'> <img width="471" alt="Screen Shot 2022-02-12 at 11 21 10 PM" src="https://user-images.githubusercontent.com/71660299/153738473-32a25afa-f337-46b9-befd-6945e3f18e4e.png">
  
  
- We shall have $n$ differenct weight vectors because we have $n$ different observations. 
**Linear regression can be seen as a linear combination of the observed outputs, or values of the target.**
    
- To get to LOWESS, we have: 
  <p align = 'center'> <img width="202" alt="Screen Shot 2022-02-12 at 11 21 35 PM" src="https://user-images.githubusercontent.com/71660299/153738477-c853bb25-d0ae-46ff-8fd6-d16012620541.png">
  
  
- We solve for *beta* (by assuming that *X^TX* is invertible): 
  <p align = 'center'> <img width="415" alt="Screen Shot 2022-02-12 at 11 22 31 PM" src="https://user-images.githubusercontent.com/71660299/153738494-34e34fe8-6d8e-4bf8-836c-36cc95c3e7aa.png">
  
- We take the expected value of this equation and obtain: 
  <p align = 'center'> <img width="222" alt="Screen Shot 2022-02-12 at 11 22 55 PM" src="https://user-images.githubusercontent.com/71660299/153738503-898e6df6-59bf-4b19-a30e-dc46c0ae5f33.png">
  
- Therefore, the predictions we make are: 
  <p align = 'center'> <img width="97" alt="Screen Shot 2022-02-12 at 11 23 32 PM" src="https://user-images.githubusercontent.com/71660299/153738521-a46d9795-d3ce-4b79-802d-408f5381a789.png">
   
    
Finally, that takes us to the locally weighted regression we have:
   <p align = 'center'> <img width="306" alt="Screen Shot 2022-02-12 at 11 19 23 PM" src="https://user-images.githubusercontent.com/71660299/153738433-1c60e39b-c996-48df-8b3d-4cea7e03a88f.png">
 
**In Locally Weighted Linear Regression, the predictions made are a linear combination of the actual observed values of the dependent variable.**
  
  
  
  
  

### Gradient Boosting
Assume you have an regressor $F$ and, for the observation $x_i$ we make the prediction $F(x_i)$. To improve the predictions, we can regard $F$ as a 'weak learner' and therefore train a decision tree (we can call it $h$) where the new output is $y_i-F(x_i)$. Thus, there are increased chances that the new regressor

$$\large F + h$$ 

is better than the old one, $F.$


## Extreme Gradient Boosting
la la la 

  
## Code and Results
  
  
  I included the K-Fold Cross Validation inside the loop to increase validity.
  
  ```markdown
 `mse_tri = []
mse_epa = []
mse_quar = []
mse_blwr = []
mse_rf = []
mse_xgb = []
mse_nn = []

for i in range(10):
  kf = KFold(n_splits=10,shuffle=True,random_state=i) #randomizing the random state through this loop
  # this is the Cross-Validation Loop
  for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)

    # LOWESS - TRICUBIC
    yhat_tri = lw_reg(xtrain,ytrain, xtest,Tricubic,tau=1.2,intercept=True)
    mse_tri.append(mse(ytest,yhat_tri))

    # LOWESS - EPANECHNIKOV
    yhat_epa = lw_reg(xtrain,ytrain, xtest,Epanechnikov,tau=1.2,intercept=True)
    mse_epa.append(mse(ytest,yhat_epa))

    # LOWESS - QUARTIC
    yhat_quar = lw_reg(xtrain,ytrain, xtest,Quartic,tau=1.2,intercept=True)
    mse_quar.append(mse(ytest,yhat_quar))

    # BOOSTED LOWESS
    yhat_blwr = boosted_lwr(xtrain,ytrain, xtest,Tricubic,tau=1.2,intercept=True)
    mse_blwr.append(mse(ytest,yhat_blwr))

    # RANDOM FOREST
    model_rf = RandomForestRegressor(n_estimators=100,max_depth=3)
    model_rf.fit(xtrain,ytrain)
    yhat_rf = model_rf.predict(xtest)
    mse_rf.append(mse(ytest,yhat_rf))

    # XGBOOST
    model_xgb.fit(xtrain,ytrain)
    yhat_xgb = model_xgb.predict(xtest)
    mse_xgb.append(mse(ytest,yhat_xgb))` 

```
  
The results were as followed: 

<p align = 'center'> <img width="455" alt="Screen Shot 2022-02-27 at 5 36 32 PM" src="https://user-images.githubusercontent.com/71660299/155902807-668ea46e-31ea-418d-a50a-656fa0fb6f84.png">




### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
