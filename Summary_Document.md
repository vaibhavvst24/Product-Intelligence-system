# Product Intelligence System

ML / GenAI Engineer Assignment



## 1\. Problem Understanding



Modern digital commerce platforms face two major AI challenges:

Limited Product Imagery

New products often lack sufficient image assets.

Marketing teams require synthetic visuals for campaigns and A/B testing.

Privacy concerns limit the use of real customer imagery.

Massive Unstructured Customer Feedback

Millions of product reviews are generated across platforms.

Manual analysis is not scalable.

Businesses require automated sentiment tracking and insight extraction.

To address these challenges, this project implements a Product Intelligence System consisting of:

A Generative Adversarial Network (GAN) for synthetic image generation.

A fine-tuned Large Language Model (LLM) for customer sentiment intelligence.



## 2\. Business Mapping



The system supports real-world business applications:

Synthetic Image Generation (GAN)

Catalog augmentation for new SKUs

Synthetic fashion/product prototypes

Marketing creative testing

Data augmentation for low-volume products

Privacy-safe synthetic assets

Review Intelligence (LLM)

Automated sentiment tracking

Product satisfaction dashboards

Voice-of-customer analytics

Brand perception monitoring

Review summarization support



Together, these components form a scalable AI-powered product intelligence framework.



## 3\. Dataset Choices



Image Dataset

Fashion MNIST

28×28 grayscale clothing images

Suitable for rapid GAN experimentation

Lightweight and GPU-friendly

Enables demonstration of adversarial learning fundamentals

Text Dataset

IMDb Reviews Dataset

50,000 labeled movie reviews

Binary sentiment classification

Balanced dataset (positive/negative)

Ideal for transformer fine-tuning demonstration



Datasets were downloaded using HuggingFace and exported locally to ensure reproducibility.



## 4\. GAN Architecture Design



A Deep Convolutional GAN (DCGAN) architecture was implemented to improve spatial feature learning compared to fully connected GANs.



Generator



Input: 100-dimensional latent noise vector

ConvTranspose layers with BatchNorm

ReLU activations

Tanh output laye

Output: 32×32 grayscale image



Discriminator



Convolutional layers

BatchNorm + LeakyReLU

Final Sigmoid output (real/fake)

Training Setup

Optimizer: Adam (lr = 0.0002, β1 = 0.5)

Epochs: 50

Loss: Binary Cross Entropy

Label smoothing applied (real = 0.9)



## 5\. LLM Fine-Tuning Strategy



Model selected:

DistilBERT



Why DistilBERT?



Lightweight and efficient

Strong baseline for classification

Suitable for limited compute environments

Fast fine-tuning capability

Fine-Tuning Pipeline

Tokenization using DistilBERT tokenizer

Train/Validation split (90/10)

HuggingFace Trainer API

Evaluation every epoch

Best model selection based on accuracy



Hyperparameters

Learning rate: 2e-5

Epochs: 3

Batch size: 16

Weight decay: 0.01



## 6\. Training Results



GAN Results



| Metric       | Result |

| ------------ | ------ |

| FID Score    | 68.2   |

| Epochs       | 50     |

| Architecture | DCGAN  |





LLM Results

Before Fine-Tuning

Accuracy ≈ ~50–60%

Near random classification performance

After Fine-Tuning

Accuracy ≈ ~90%

F1 Score ≈ ~90%

ROUGE-L improved significantly



Fine-tuning demonstrated strong domain adaptation and task-specific learning.



## 7\. Evaluation Metrics

GAN Evaluation



Visual inspection across epochs

Loss convergence behavior

Mode collapse monitoring

Fréchet Inception Distance (FID = 68.2)

LLM Evaluation

Accuracy

F1 Score

ROUGE-L

Validation per epoch

Before vs After comparison

Both quantitative and qualitative metrics were used to validate performance.



Sample Outputs

GAN

Synthetic clothing samples

Progressive improvement across epochs

Diverse shape generation

LLM



Example Prediction:



Input:



"This product was absolutely amazing!"



{

&nbsp; "prediction": "Positive",

&nbsp; "confidence": 0.9842

}



## System Design \& Modularity



The system was packaged into reusable modules: Inputs, Image dataset, Review dataset, Processing Pipelines, GAN training module, LLM fine-tuning module, Inference module



Outputs



Synthetic images, Trained GAN model, Fine-tuned DistilBERT model, Sentiment prediction outputs



### Limitations \& Improvements



GAN Limitations



Low-resolution output (32×32)

FID of 68.2 indicates room for realism improvement

Limited dataset diversity



### Improvements



WGAN-GP for improved stability

Higher resolution datasets (e.g., DeepFashion Dataset)

Progressive growing GANs

Style-based generators

LLM Limitations

Binary sentiment only

No domain-specific product vocabulary adaptation

No parameter-efficient fine-tuning (e.g., LoRA)



### Conclusion

#### 

This project demonstrates:



Adversarial training understanding (GAN)

Transformer fine-tuning workflows (LLM)

Quantitative evaluation (FID, Accuracy, F1, ROUGE)

Modular engineering design

Production-oriented inference pipeline



The implemented Product Intelligence System provides a scalable foundation for synthetic asset generation and automated customer insight analysis in digital commerce environments.

