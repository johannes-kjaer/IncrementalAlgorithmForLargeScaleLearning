# An Incremental Algorithm for Large Scale Learning

This project aims to explore the dual coordinate descent algorithms proposed by Yu, Huang, and Lin in their paper 'Dual Coordinate Descent Methods for Logistic Regression (LR) and Maximum Entropy (ME) Models', available here: https://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf

The algorithms aims to improve computational efficiency for the dual problem of Maximum Entropy and Logistic Regression, multi-class and binary classification, respectively. 

In this project we implement the algorithms, and compare them to similar algorithms learned in the course MATH-412, Statistical Machine Learning at EPFL.

## Overview of the algorithms

The algorithms aim to solve the dual problem of LR and ME, which might yield a faster convergence. Then a coordinate descent method is used. That involves selecting one variable at a time, which creates a sub-problem which is solved using a modified Newton method to minimize with respect to the chosen variable. 

To brace for numerical instabilities the algorithm operates with some variables constructed from the problem variables. These are more numerically stable. Afterwards the relevant problem parameters are refound from the constructed parameters.