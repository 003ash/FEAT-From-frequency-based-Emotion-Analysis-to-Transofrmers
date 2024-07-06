# FEAT: From Frequency-based Emotion Analysis to Transformers

## Project Overview

This project, titled "FEAT: From Frequency-based Emotion Analysis to Transformers," explores various approaches to emotion classification in text data. We investigate the effectiveness of traditional machine learning models, transformer-based models, and hybrid methods that combine the strengths of both.

## Objectives

1. Implement and evaluate a baseline transformer model for emotion classification.
2. Explore the effectiveness of traditional machine learning models using features extracted from transformer models.
3. Investigate the impact of fusing TF-IDF features with transformer-based embeddings.
4. Develop and assess an improved transformer model with additional layers for feature extraction.
5. Compare the performance of different approaches across various evaluation metrics.

## Dataset

We used the "dair-ai/emotion" dataset from Hugging Face, which contains text data labeled with six emotion categories.

## Models and Approaches

1. **Baseline Transformer Model**: A DistilRoBERTa-based model fine-tuned for emotion classification.
2. **Traditional ML Models**: SVC, Random Forest, Gradient Boosting, and AdaBoost classifiers.
3. **Hybrid Approach**: Combining transformer embeddings with TF-IDF features.
4. **Improved Transformer Model**: An enhanced version of the baseline with additional layers for feature extraction.
5. **Ensemble Model**: A combination of DistilRoBERTa and DistilBERT models.

## Key Components

1. `baseline_and_improvements.py`: Contains the implementation of baseline and improved transformer models, as well as the hybrid approach.
2. `ensemble.py`: Implements the ensemble model combining two transformer architectures.

## Results and Outcomes

1. The baseline transformer model (EmotionClassifier) achieved competitive performance in emotion classification.
2. Traditional ML models trained on features extracted from the fine-tuned transformer showed promising results, with some outperforming the baseline in certain metrics.
3. Fusing TF-IDF features with transformer embeddings led to improved performance for some ML models, demonstrating the potential of combining different feature representations.
4. The improved transformer model (EmotionClassifierImproved) with additional layers showed enhanced performance compared to the baseline.
5. The ensemble approach combining DistilRoBERTa and DistilBERT demonstrated the highest overall performance, leveraging the strengths of both models.

## Key Findings

1. Transformer-based models provide a strong foundation for emotion classification tasks.
2. Traditional ML models can effectively utilize features extracted from transformers, offering a balance between performance and computational efficiency.
3. Fusing different types of features (e.g., transformer embeddings and TF-IDF) can lead to improved performance in some cases.
4. Ensemble approaches combining multiple transformer architectures show promise for achieving state-of-the-art results.

## Setup and Usage

1. Install required dependencies:

  ```pip install torch transformers datasets evaluate scikit-learn tqdm```

2. Run the baseline and improvements:

  ```python baseline_and_improvements.py```

3. Run the ensemble model:

  ```python ensemble.py```

## Future Work

1. Explore other transformer architectures and pre-trained models for emotion classification.
2. Investigate more sophisticated feature fusion techniques.
3. Experiment with multi-task learning approaches to leverage related NLP tasks.
4. Analyze model performance on different emotion categories and identify areas for improvement.
5. Explore the impact of different hyperparameters and training strategies on model performance.

## References

1. Saravia, E., et al. (2018). CARER: Contextualized affect representations for emotion recognition. Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing.
2. Talaat, A. S. (2023). Sentiment analysis classification system using hybrid BERT models. Journal of Big Data, 10(1), 110.
