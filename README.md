# Automatic Ticket Classification - NLP with NMF and ML Models

## Table of Contents
* [Topic Modelling using NMF Overview](#topic-modelling-using-nmf-overview)
* [Problem Statement](#problem-statement)
* [Technologies Used](#technologies-used)
* [Approach for Modeling](#approach-for-modeling)
* [Classification Outcome](#classification-outcome)
* [Conclusion](#conclusion)
* [Acknowledgements](#acknowledgements)

## Topic Modelling using NMF Overview

**Topic modeling** is an unsupervised machine learning technique used to discover hidden thematic structures in a collection of documents. **Non-Negative Matrix Factorization (NMF)** is a popular algorithm for topic modeling, as it decomposes the document-term matrix into two lower-dimensional matrices, representing the topic distributions for documents and the term distributions for topics. By extracting these topics, we can cluster documents into meaningful categories and use this labeled data to train supervised models for further applications.

In this exercise, NMF was used to classify customer support tickets into five meaningful topics. These topics were then used to train machine learning models to automate the process of ticket categorization.

---

## Problem Statement

The manual classification of customer support tickets is a time-consuming and error-prone process. This project aims to develop an automated ticket classification system that uses topic modeling and machine learning to categorize tickets into predefined topics. The primary goals are:

- To extract meaningful topics from customer support tickets using NMF.
- To create labeled data based on the extracted topics.
- To train machine learning models on the labeled data for automatic ticket classification.

This solution seeks to improve operational efficiency by reducing manual effort and increasing the accuracy of ticket categorization.

---

## Technologies Used

The following technologies and libraries were used in this exercise:

- **Python**: Programming language for building and testing models.
- **Scikit-learn**: For NMF, vectorization, and machine learning models.
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical computations.
- **Matplotlib/Seaborn**: For data visualization.
- **Jupyter Notebook**: For interactive development and experimentation.

---

## Approach for Modeling

1. **Data Preprocessing**:
   - Cleaned and prepared the text data by removing stop words, punctuation, and performing lemmatization.
   - Transformed the text data into a document-term matrix using TF-IDF vectorization.

2. **Topic Modeling with NMF**:
   - Applied the NMF algorithm to extract five distinct topics from the text data.
   - Reviewed the top keywords for each topic and labeled the data accordingly.

3. **Supervised Classification**:
   - Created labeled datasets using the NMF topics as categories.
   - Trained four machine learning models: Logistic Regression, Decision Tree, Random Forest, and Naive Bayes.
   - Evaluated models on metrics such as accuracy, precision, recall etc.

4. **Model Selection**:
   - Logistic Regression emerged as the best-performing model, achieving an accuracy of 91% on the test dataset.

---

## Classification Outcome

The labeled dataset created from NMF-based topic modeling was used to train and evaluate machine learning models for automatic ticket classification. The key results are:

- **Best Model**: Logistic Regression
- **Test Accuracy**: 91%
- **Balanced Precision and Recall**: Achieved consistent performance across all five categories.

The Logistic Regression model demonstrated the ability to classify tickets with high accuracy and robustness, making it the ideal choice for this task.

---

## Conclusion

This project highlights the effectiveness of combining unsupervised topic modeling with supervised machine learning for automating ticket classification. NMF provided meaningful insights into the thematic structure of tickets, while the Logistic Regression model delivered high accuracy in classification. By automating the ticket categorization process, this solution significantly reduces manual effort and improves operational efficiency. The approach demonstrated here can be extended to similar classification problems in various domains.

---

## Acknowledgements

This case study has been developed as part of Post Graduate Diploma Program on Machine Learning and AI, offered jointly by Indian Institute of Information Technology, Bangalore (IIIT-B) and upGrad.