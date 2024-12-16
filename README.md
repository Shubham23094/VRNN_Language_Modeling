# Variational Recurrent Neural Network (VRNN) for Language Modeling

This repository implements a **Variational Recurrent Neural Network (VRNN)** for language modeling and sentence classification tasks. The VRNN combines **Variational Autoencoders (VAE)** with **Recurrent Neural Networks (RNN)** to enhance both classification performance and uncertainty estimation in sequence data.

The project compares VRNN with Simple RNN and Bayesian RNN models in two classification tasks: **Sentiment Analysis** and **Spam Classification**.

---

## Project Overview

The **VRNN** leverages variational dropout to regularize the network, reducing overfitting and improving generalization. It combines the strengths of RNNs for sequential data modeling and VAEs for uncertainty estimation. The model has been evaluated on two datasets:

- **SMS Spam Collection Dataset**: Contains 5,574 SMS messages, categorized into "ham" (non-spam) and "spam" labels.
- **Sentiment Analysis Dataset**: Contains tweets labeled as "positive" or "negative" based on sentiment.

## Key Features

- **VRNN with Variational Dropout**: Outperforms Simple RNN and Bayesian RNN models in classification accuracy and uncertainty estimation.
- **Uncertainty Quantification**: Helps in decision-making when prediction confidence is important.
- **Datasets**: Two datasets for text classification (Sentiment Analysis and Spam Classification).
  
## Files in this Repository

### Python Scripts

1. **`simple_rnn.py`**  
   Implementation of a Simple Recurrent Neural Network (RNN) for sequence classification.  
   Usage: `python simple_rnn.py`

2. **`vrnn_with_dropout.py`**  
   Implementation of the VRNN model with variational dropout. This model utilizes dropout regularization for better generalization.  
   Usage: `python vrnn_with_dropout.py`

3. **`vrnn_without_dropout.py`**  
   Implementation of the VRNN model without variational dropout for comparison purposes.  
   Usage: `python vrnn_without_dropout.py`

### Datasets

- **SMS Spam Collection Dataset**: The dataset contains 5,574 SMS messages, categorized as "spam" or "ham".  
- **Sentiment Analysis Dataset**: Contains 1.6 million labeled tweets for sentiment classification ("positive" or "negative").

Dataset can be accessed from below link : https://drive.google.com/drive/folders/1qkDN62pPlfKMzMN39iAnR9h0XKqehqeK?usp=sharing
### Results

Model performance and evaluation results, including accuracy, precision, recall, and F1-score, can be found in the report. The VRNN model with variational dropout achieved an accuracy of **88.45%** on Sentiment Analysis and **97.67%** on Spam Classification.

## Installation

To install and run the project, make sure to have the following dependencies:

1. **Python 3.7+**
2. Install the required Python libraries:


## Requirements
The required libraries for running the scripts and models are:

torch (PyTorch)
numpy
pandas
scikit-learn
matplotlib
tensorflow (if using any dependencies for RNNs)
tqdm (for progress bars during training)
A requirements.txt file is provided with the necessary libraries.

## Usage
Training the VRNN Model
To train the VRNN model with variational dropout:
python vrnn_with_dropout.py
To train the VRNN model without variational dropout:
python vrnn_without_dropout.py
To train the Simple RNN model:
python simple_rnn.py
Sampling Predictions with Uncertainty
For uncertainty-aware predictions (VRNN with variational dropout), use the posterior sampling technique provided in vrnn_with_dropout.py to get both predicted labels and uncertainty estimates.

## Results
### Model Performance (Accuracy) on Sentiment Analysis and Spam Classification Datasets

| Model                             | Sentiment Analysis Accuracy (%) | Spam Classification Accuracy (%) |
|-----------------------------------|---------------------------------|----------------------------------|
| VRNN with Variational Dropout     | 88.45%                          | 97.67%                           |
| VRNN without Variational Dropout  | 87.50%                          | 97.67%                           |
| Simple RNN                        | 88.38%                          | 97.49%                           |

## Conclusion
This repository presents a novel Variational Recurrent Neural Network (VRNN) architecture, combining the benefits of variational autoencoders and recurrent networks for improved text classification and uncertainty estimation. The model's performance on both sentiment analysis and spam classification demonstrates its ability to model uncertainty effectively while achieving competitive classification accuracy.

Future work will focus on applying VRNN to other NLP tasks like machine translation and text generation. Enhanced uncertainty estimation will be explored in active learning and out-of-domain detection tasks.

## References
Variational Recurrent Neural Networks (VRNN)
Original research paper and model implementation details can be found in the project report.


