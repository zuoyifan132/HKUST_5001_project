# <center>MSBD 5001 Final Report<center/>

##Introduction:

---

Our question for this project is: what is the best model to predict online shoppers' behavior base on existing data and what is the main factors or features impact their final decision? The dataset that we used has over 12,000 records of difference users in 1 year period (84.5% negative). The features contains ten numerical and eight categorical variables. We tried four different models: Logistic Regression model done by WIllian; Neural Network model done by Evan; KNN and random forest model done by Jerry. The best model is voted by the F-score of all models. Feature importance envolve only three models(Logistic, Neural Network and Random Forest) by different methods.

###Design and Implementation:

---

- MLPClassifier:

  - Solver: ``Adam``

    |  Solver  | lbfgs  |  sgd  | adam  |
    | :------: | :----: | :---: | :---: |
    | Accuracy | 88.77% | 89.1% | 90.1% |

    We tried all three solver, we found that the adam achieve the best performance which is 90.1% since it combines the adaptive learning rate and the moments.

  - Hidden layer size: ``(6,3)``

    - Number of perceptrons: (we fix the second layer to be 3)

      | #perceptrons |   3    |     6     |   10   |   20   |   30   |
      | :----------: | :----: | :-------: | :----: | :----: | :----: |
      |   Accuracy   | 88.12% | **89.1%** | 89.07% | 88.14% | 87.51% |

      We see that through the experiments, less than three perceptrons layer drops the accuracy compare to six perceptrons since too less perceptrons cannot represent the function of the dataset then the model is underfitting the data. Larger than 6 perceptrons also drops the accuracy compare to six perceptrons, becasue too much perceptrons overfitting the data. 

    - Number of layers: (we fix the number of perceptrons to be 6)

      | #layers  |   1    |     2     |   3    |   5    |   10   |
      | :------: | :----: | :-------: | :----: | :----: | :----: |
      | Accuracy | 88.65% | **89.1%** | 88.94% | 87.56% | 87.01% |

      By the experiments, the result show that the best performance accuracy which is 89.1% is provide by the two layers network. Too less number of layer decrease the accuracy since it cannot represent complex model and too much number of layer also decrease the accuracy which is due to the difficulty of gradients to propagate back to the lower layer.

  - Activation function: ``Relu``

    Relu is commonly used in the hidden layer as the activation function.

  - Max iteration: ``1000``

    Since our model cannot converge within default 200 iteration, so we set the max iteration to be 1000 in order to let the model converge.

  - Alpha: ``0.0001``

    We use the default value for the strength of L2 regularization term.

### Permutation feature Importance

---

Permutation feature importance measures the increase in the prediction error of the model after we permuted the feature’s values, which breaks the relationship between the feature and the true outcome.

- Theory:

  - A feature is consider significant if shuffling its values increases the model error, because in this case the model relied on the feature for the prediction.
  - A feature is consider non-significant if shuffling its values leaves the model error unchanged, because in this case the model ignored the feature for the prediction. 

- Algorithm:

  <img src="/Users/sakazuho/Desktop/HKUST/first_term/5001/project/report/Screen Shot 2022-11-17 at 9.49.39 PM.png" style="zoom:50%;" />

- For our dataset:

  - Original model error: **0.09813**

  - Transformation:

    - features_error = (features_error - original_error) / original_error
    - It means that How much percent each feature error exceed the original error 

  - Each feature error after the transformation:

    <img src="/Users/sakazuho/Desktop/HKUST/first_term/5001/msbd5001-project/report source/zuoyifan/Screen Shot 2022-11-12 at 3.53.56 PM.png" alt="Screen Shot 2022-11-12 at 3.53.56 PM" style="zoom:50%;" />

  - The result shows that the most significant feature is pagevalue, this feature error exceeding the original error by 60.7%. This means that the pagevalue has really big impact on the result of transaction.

  - The top five features siginificant features:

    - PageValue
    - Month_Nov
    - informational duration
    - Month_Dec
    - Administrative duration. 