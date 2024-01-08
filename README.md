[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mXsE_ooTJyO3xl99o6Oyp8P2mbXOD638#scrollTo=_S3iN__8DgNg)
 [![Snowflake](https://img.shields.io/badge/Snowflake-Ready-brightgreen)](https://www.snowflake.com/)
# NLP-analysis
This project comprises three interconnected phases, collectively illuminating the landscape of natural language processing and artificial intelligence. The first phase, "Sentence Classification," scrutinizes text data at a granular level, identifying the presence of sexist content using models such as "bert-base-uncased," "xlbert-large-uncased," and "roberta-large." 
This initial phase sets the stage for the second, "Explainable Algorithm Development," which seeks to demystify the intricate workings of AI models, including "xlbert-large-uncased," in recognizing sexist tokens. 
The final phase, "Target Identification," expands the horizon by pinpointing and elucidating specific targets of sexist comments. Together, these phases endeavor to bolster the transparency and effectiveness of AI models in textual analysis.

![Analysis Process](https://github.com/msbeigi/NLP-analysis/blob/main/img/process-layout.jpg)

## Exploring Word Frequency 
Words are displayed in a specific word size based on their frequency in the dataset:

![Analysis Process](https://github.com/msbeigi/NLP-analysis/blob/main/img/word-freq.png)

## Utilizing Explainable Algorithms: LIME and SHAP
Effectively highlight important tokens in identifying sexist content across various examples using LIME and SHAP.

### Token Detection
Compare the total tokens detected by LIME and SHAP.

![Analysis Process](https://github.com/msbeigi/NLP-analysis/blob/main/img/Total%20Tokens%20Comparison%20of%20LIME%20and%20SHAP.png)

### Target Detection
Compare the total targets detected by LIME and SHAP.
![Analysis Process](https://github.com/msbeigi/NLP-analysis/blob/main/img/Total%20Target%20Comparison%20of%20LIME%20and%20SHAP.png)
