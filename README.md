# prediction-of-diabetic-level <br>

### Dataset: <br>
The pickle file "diabetes.pickle" contains medical dataset along with the target assignment. The input
variables correspond to measurements (physical, physiological, and blood related) for a given patient
and the target variable corresponds to the level of diabetic condition in the patient. It contains: <br>
xtrain (242 x 64) and ytrain (242 x 1) for training. <br>
xtest (200 x 64) and ytest (200 x 1) for testing. <br>

### Approach 1: Linear Regression <br>
MSE without intercept is higher than MSE with intercept. Without considering the intercept,
the line is forced to pass through the origin which causes greater difference between actual
values and estimated values. When intercept is considered, this difference is minimized and
hence MSE is lower. <br>
##### Observations:
MSE without intercept for test data 106775.361555 <br>
MSE with intercept for test data 3707.84018132 <br>
MSE without intercept for train data 19099.4468446 <br>
MSE with intercept for train data 2187.16029493 <br>

### Approach 2: Ridge Regression <br>
##### Comparison of weights:<br>
The L2 norm values of the weights learnt is as follows: <br>
Using Linear Regression: 1.55081011e+10<br>
Using Ridge Regression: 920281.35693682<br>
##### Comparison of mean square errors:<br>
Both the approaches behave closely in terms of errors on test and train data.
MSE using ridge regression:<br>
Train data:2451.52849064<br>
Test data:2851.33021344<br>
MSE using linear regression:<br>
Train data:2187.16029493<br>
Test data:3707.84018132<br>
##### Optimum value of lambda: 0.06 <br>
The optimal value is determined by observing the MSE on test data at each value of lambda.
Lambda is a regularization parameter which is used to avoid the problem of overfitting on
train data. Hence, the value where the MSE on test data is lowest is considered as the
optimum value, since it represents best performance on the test data.<br>

### Approach 3: Gradient Descent for Ridge Regression <br>
The error curve obtained by gradient descent follows the curve obtained by matrix inverse
method (used in Approach 2) closely. These two curves overlaps as the number of iterations
are increased. <br>

### Approach 4: Non-linear Regression<br>
After plotting the graphs for values of p, it is clear that as p(dimension) goes on increasing, the line is overfitted over
the train data and hence we see an increase in the MSE on the test data. In case of regularization, it controls the overfitting 
and hence we see an almost constant value of MSE for higher values of p. <br>

##### Optimal value of p:
Lambda =0 implies that there is no regularization.Hence, we observe the lowest MSE at p=1 <br>
Lambda = 0.06: In this case the MSE is almost constant for values higher than 1. However, the lowest value
is obtained at p=4. <br>

### Comparison of approaches in terms of training and test errors:
The minimum error value (i.e. considering intercept) on training and test data using linear
regression is 2187 and 3707 respectively.<br>
The minimum error value (for lambda = 0.06) on training and test data using ridge regression
is 2451 and 2851 respectively. <br>
The minimum error values (for lambda = 0.06) on training and test data using non-linear
regression are around 3900 which are higher than both the above mentioned methods.<br>
The error values on train data for linear and ridge regression are close to each other.
Additionally, the performance of method is mainly determined by error value on unknown
test data. Hence, ridge regression performs better than linear regression on test data. For
various values of regularization parameter minimum mean squared error can be determined
which can be used to choose the best setting.

