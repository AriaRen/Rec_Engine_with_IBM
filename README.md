# Recommendation with IBM Watson Studio
This project aims to make article recommendations for IBM Watson Studio's data platform based off of historical user interactions. 

## Table of Contents
1. Installation
2. Project Motivation
3. File Descriptions
4. Results
5. Licensing, Authors, and Acknowledgements

## installation
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.

## Project Motivation
For this project, I was interested in making recommendations by different methods to satisfy users in different types. 

1. Rank-based Recommendations: used for absolute new users who had no historical interactions with platform.
2. User-based Collaborative: used for old users who already had some article preference.
3. Matrix Factorization: Machine Learning methods to provide user preference predictions even for new users. Also to explore how the number of latent factors could affect prediction.

## File Description
There is 1 python script available here to showcase work related to the above methods. All of the questions are solved in sequence of above list. The script includes data and package loading, exploratory analysis, data cleaning, transformation, coding for different recommendation methods and whole process of building predictive models.

## Results
The different recommendation engines are all stored in corresponding functions in script. 
