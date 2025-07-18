# Suicidal Detection OnSIDe-Bert-CNN-BERT-CNN-

This repository contains the code and resources for a project focused on detecting suicidal ideation in text using a hybrid deep learning model, OnSIDe-Bert-CNN. This model combines BERT (Bidirectional Encoder Representations from Transformers) for contextual embedding and Convolutional Neural Networks (CNNs) for feature extraction and classification.

## Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Preprocessing](#preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Sentiment Analysis](#sentiment-analysis)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## About the Project

This project aims to develop an effective and robust model for detecting suicidal ideation in text data. The model utilizes a hybrid architecture, OnSIDe-BERT-CNN, leveraging the strengths of both BERT and CNNs to achieve high accuracy in sentiment classification. This project is intended for research and educational purposes, specifically in the field of AI Applications for mental health.

## Datasets

We used two datasets:

1.  **Reddit Dataset:** Used for training, validation, and testing. Scraped from the SuicideWatch and teenagers subreddits. Available on Kaggle: [Suicide Watch Dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch).
2.  **Twitter Dataset:** Used for final model testing on unseen data. Available on GitHub: [Twitter Suicidal Intention Dataset](https://github.com/laxmimerit/twitter-suicidal-intention-dataset).

## Dependencies

To run this project, you will need the following Python libraries:

-   `pandas`
-   `numpy`
-   `spacy`
-   `unidecode`
-   `contractions`
-   `re`
-   `wordninja`
-   `collections`
-   `pkg_resources`
-   `spellchecker`
-   `symspellpy`
-   `matplotlib`
-   `seaborn`
-   `nltk`
-   `empath`
-   `vaderSentiment`
-   `transformers`
-   `torch`
-   `scikit-learn`

You can install these dependencies using pip:

```
pip install pandas numpy spacy unidecode contractions wordninja spellchecker symspellpy matplotlib seaborn nltk empath vaderSentiment transformers torch scikit-learn
```

## Installation
1.Clone the repository:
```
git clone [https://github.com/DeepigaLoganathamoorthy/Suicidal_Detection_OnSIDe-Bert-CNN-BERT-CNN-.git](https://www.google.com/search q=https://github.com/DeepigaLoganathamoorthy/Suicidal_Detection_OnSIDe-Bert-CNN-BERT-CNN-.git)
cd Suicidal_Detection_OnSIDe-Bert-CNN-BERT-CNN-
```

2. Install the required dependencies (see Dependencies).
3. Ensure the dataset downloaded from the source(s) and placed in your data/ folder.


## Repository Structure
Suicidal_Detection_OnSIDe-Bert-CNN/
│
├── data/                          # Folder to store raw and cleaned datasets
│   ├── suicide_final_cleaned.csv
│
├── models/                        # Saved models
│   ├── bert_cnn_model.pth
│
├── src/                           # Source code files (if separated)
│   ├── preprocessing.py
│   ├── eda.py
│   ├── sentiment_analysis.py
│   ├── model_bert.py
│   ├── model_cnn.py
│   ├── model_onside.py
│   ├── train.py
│   ├── evaluate.py
│
├── utils/                         # Utility functions (optional)
│
├── notebooks/                     # Jupyter notebooks for exploration
│
├── requirements.txt               # Dependency list (recommended addition)
├── README.md                      # Project overview and usage instructions

## How to Run the Model
Once you've cloned the repo and installed dependencies:
1.Download & Prepare Datasets:
- Download the datasets from the links provided in the README.
- Place them in the data/ folder.
2.Run Preprocessing:
- python src/preprocessing.py
3. Perform EDA:
- python src/eda.py
4. Run Sentiment Analysis:
- python src/sentiment_analysis.py
5. Train the Model:
- python src/train.py --model onside
6. Evaluate the Model:
- python src/evaluate.py --model onside

--model can be set to bert, cnn, or onside depending on what you want to train/evaluate.


## Usage

1.  **Preprocessing:**
    * Run the preprocessing script to clean and prepare the data.
    * The cleaning and preprocessing code is located at the top of the python file.
    * The cleaned dataset is saved as `suicide_final_cleaned.csv`.

2.  **Exploratory Data Analysis (EDA):**
    * Run the EDA scripts to visualize and analyze the data.
    * EDA code is included in the python file.

3.  **Sentiment Analysis:**
    * Run the sentiment analysis scripts for Empath and VADER analysis.
    * Sentiment analysis code is included in the python file.

4.  **Model Training:**
    * Run the model training scripts for BERT, CNN, and OnSIDe-BERT-CNN models.
    * Model training code for BERT, CNN, and (hybrid) OnSIDe-BERT-CNN is included in the python file.
    * The OnSIDe-BERT-CNN model is implemented as a PyTorch `nn.Module`.
        * **BERT Embedding:** The `bert-base-uncased` model is used to generate contextualized word embeddings.
        * **CNN Layers:** Multiple 2D convolutional layers with varying filter sizes (`[3, 4, 5]`) are applied to the BERT embeddings. ReLU activation and max-pooling are used for feature extraction.
        * **Fully Connected Layer:** The extracted features are flattened, passed through a dropout layer, and then fed into a fully connected layer with a sigmoid activation function for binary classification.
    * **Training Process:**
        * **Initialization:** The model, optimizer (AdamW), and loss function (BCEWithLogitsLoss) are initialized.
        * **Epochs:** The model is trained for a specified number of epochs (`EPOCHS = 1`).
        * **Batch Training:** In each epoch, the training data is iterated in batches.
        * **Forward Pass:** For each batch, the input sequences are passed through the model to obtain predictions.
        * **Loss Calculation:** The loss between the predicted and actual labels is calculated.
        * **Backward Pass and Optimization:** The loss is backpropagated, and the model's parameters are updated using the optimizer.
        * **Loss Tracking:** The average training loss is calculated and printed for each epoch.
        * **Model Saving:** After training, the trained model's state dictionary is saved to `bert_cnn_model.pth`.

5.  **Evaluation:**
    * The evaluation metrics and confusion matrices are displayed after each model training.
    * Models are evaluated using accuracy, precision, recall, F1-score, and confusion matrices.

## Model Architecture

   ![image](https://github.com/user-attachments/assets/d3b730d5-a909-4c7a-9f2e-6ac1d56ba683)
   
1. BERT Model: Uses the bert-base-uncased pre-trained model for sequence classification.
2. CNN Model: A convolutional neural network model with embedding, convolutional, and dense layers.
3. OnSIDe-Bert-CNN Model: A hybrid model combining BERT embeddings with CNN layers for feature extraction and classification.

![image](https://github.com/user-attachments/assets/96eb3257-9499-4f5d-a317-5fee8ad132b4)


## Preprocessing
The preprocessing steps include:
a) Removing extra whitespaces
b) Removing accented characters
c) Removing URLs
d) Removing symbols and digits
e) Removing special characters
f) Fixing word lengthening
g) Expanding contractions
h) Lowercasing text
i) Removing stop words
j) Lemmatization
k) Spelling correction

## Exploratory Data Analysis (EDA)
The EDA includes:
1. Word count distribution for suicidal and non-suicidal texts.

![image](https://github.com/user-attachments/assets/ead8771c-b661-46d5-8130-aeb3245f663c)

2. Top bi-grams for suicidal and non-suicidal texts
3. Text length distribution

## Sentiment Analysis
1. Empath Analysis: Analyzes the text for categories like sadness, anger, and fear.
2. VADER Analysis: Analyzes the text for positive, neutral, and negative sentiment scores.

<img src="https://github.com/user-attachments/assets/e28fb16a-1f3f-4297-914c-7b02d6860d80" width="400">

## Model Training
1. BERT Model: Fine-tuned the pre-trained BERT model using the training data.
2. CNN Model: Trained a CNN model using word embeddings and convolutional layers.
3. OnSIDe-Bert-CNN Model: Trained a hybrid model combining BERT embeddings with CNN layers.

## Evaluation
The models are evaluated using the following metrics:
a) Accuracy
b) Precision
c) Recall
d) F1-score

## Results
1. The evaluation results are displayed after each model training.
2. The below table shows the model's performance trained and tested from Reddit dataset.
   
<img src="https://github.com/user-attachments/assets/dbc8e8f8-62b9-4fb4-aeef-73be4ef44a04" width="400">

3. The final model performance for Twitter dataset shows as below. The OnSIDe-Bert-CNN model is expected to achieve high accuracy in detecting suicidal ideation.

<img src="https://github.com/user-attachments/assets/f4c7d0fa-11e1-4ccc-ac53-e25a07097ded" width="400">

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues to improve this project.
