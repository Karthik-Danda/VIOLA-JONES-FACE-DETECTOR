# VIOLA-JONES-FACE-DETECTOR
A Face classifier using viola jones algorithm.

This project is an implementation of the viola jones face detector and to demonstrate howe fast it is at detecting the faces.

The main adadvantage of this algorithm is there is no need for the GPU and can be locally run any standard PC.

We apply haar filters as defined in the original viola jones paper for the feature extraction. the total number of features extracted for an image is ~ 136,000.

Each feature acts as a week classifier. 

We use a adaboost to linearly combine all the weak classifier into a strong classifier. 

The data set I have used is of size 1400 image 500 for the face and 500 for the non face.

Eventhough the classifier is insanely fast(~0.74 Sec) the max efficiency is below 90%(common for every viola jones classifier). Hence I have rigged the classifier to give more false positives. 

If there is face in an image there is a 97% probability of classifying it as a face.

The total processing time of the algorithm is around 150 min. 110 min for training and 40 min for feature extraction.

