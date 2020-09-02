# Analysis-of-ML-algorithms-for-accurate-Breast-Cancer-Classification
This project is intended to apply various different machine learning algorithms for breast cancer classification. 
First, we choose a Wisconsin Breast Cancer dataset which contains breast cancer cell features such as compactness,
radius. 
Next, we analyzed this dataset, checked for null values, replaced categorical data and we also scale the values
to prepare the data.
Once data is prepared, we successfully implemented SVM, logistic regression, k-NN, naive Bayes, decision tree,
random forest, Neural Network. 
For each of the algorithms, we used 5 fold validation. 
For each fold, we trained the model and obtained predictions on the test data, then we plot the ROC curve and PR curve, area under the curve,
determine the balance accuracy. 
Once all 5 folds are completed, we also plot the mean ROC curve, mean PR curve,
the mean area under the curve and mean accuracy. We also compare the accuracy of all the algorithms applied to
determine the best fitting algorithm for our requirements.


# Dataset Information:
The dataset used in this Project is “Wisconsin Breast Cancer Dataset”. This dataset is publicly available and was
created by Dr. William H. Wolberg, a physician at the University Of Wisconsin Hospital at Madison, Wisconsin,
USA. For this project, the dataset is a ‘.csv’ file and it was downloaded from Kaggel[1].

 This dataset was created by Dr. Wolberg who used fluid samples, taken from patients with solid breast masses and
 an easy-to-use graphical computer program called Xcyt.
 This program uses a curve-fitting algorithm, to compute ten features from each one of the cells in the sample, then it
 calculates the mean value, extreme value and standard error of each feature for the image, returning a 30 real-valued
 vector.
 
 # Instructions to Run:
 1.) GitClone this Repository.
 
 2.) Follow the instructions exactly as provided in the Read-me File.
 
 3.) This code is open source and can be used by anyone for Research purposes, but please do cite it.
 
 4.) For any additional queries, contact rakesh.nagaraju@sjsu.edu or rakenju@gmail.com.
