# Data Science
This repository contains implementation of Weed Classification system which uses CNN technique to recognise the kind of weed.

Problem Statement: Summary Statistics and Data Visualization

Description: A Deep learning system that predicts the kind of the weed you upload as an image, and gives the kind of the weed as the answer, it uses a sequential CNN model for single input and single output purposes.

Methods: Utilize Python libraries such as Tensorflow to process the image and convert images into a tensor for the model to understand,
Pandas for data manipulation, 
Matplotlib/Seaborn for data visualization, 
and NumPy for numerical operations,
scikit learn for providing a classification report for the overall performance of the model,
streamlit to make a website using python for a FrontEnd user interface

How to use:

- Just Provide a well structured dataset of weed images, structured into 'train', 'test', 'valid' sets
- train the model using this data
- analyse the performance of the model based upon the data and the layering of the model, adjust the size of the model based upon your data
- Save the model in the same directory
- Run the main.py by running the command 'streamlit run main.py'