m3.py is the main deep learning model implemented from scratch without the use of any library using concepts like forward propogation , backward propogation and gradient descent . The network consists of a 3 hidden layers with 256 , 128 and 64 neurons each.

The feature_extraction.py file extracts the features for each plate in the data which are to be used for predicting a plates's price in an array of length 420 and stores it as a pickle file. The working for the same is mentioned below.

This is a solution to this competition on kaggle --> https://www.kaggle.com/competitions/russian-car-plates-prices-prediction
Number plates can be both 8 or 9 digit long -->  X780PC797 or X001EH36 , the last 3 digits represent the region car belongs to and some combinations of the 3 alphabets in the number plate correspond to some special government codes which means the car will get some extra privilleges thus raising the plate's price.
For the features each alpha-numerical is hot encoded to a vector in a 37 dimensional space (array of length 37), for example the letter "A" gets the vector [1,0,0,0.....0] "B" gets [0,1,0,0....0].
The more recently the number plate is purchased, higher will be its cost because of inflation. So another feature is the difference in the dates between the first plate bought and the date for any given plate for which the feature has to be extracted.
Lastly the privilleges given to government codes are listen in a dictionary in the supplementary_english.py file in numerical format only , so there is no need to manipulate that data and it can be used as a feature as it is.

