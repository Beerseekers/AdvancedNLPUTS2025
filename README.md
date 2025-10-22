# AdvancedNLPUTS2025
Detecting AI-Generated Text with Transformer Fine-Tuning
This project explores multiple transformer-based approaches for detecting AI-generated
text, integrating fine-tuning and model distillation within a unified NLP pipeline. Using the public AI vs Human Text dataset, which contains over 480,000 labeled samples, we
train and evaluate three complementary architectures: MiniDeBERTa (microsoft/deberta-
v3-small), DistilBERT (distilbert-base-uncased), and a full BERT-base (bert-base-uncased)
model whose knowledge is subsequently distilled into a lightweight student network. The
workflow includes text preprocessing, stratified data splitting, and fine-tuning with the
AdamW optimizer and a linear learning-rate scheduler. Experiments were executed on
Google Colab (T4 GPU) using a 1% subset of the dataset for efficiency.
The three methods represent different strategies for balancing computational cost,
representational depth, and deployment readiness. MiniDeBERTa leverages disentangled
attention to achieve strong contextual modeling at reduced size; DistilBERT provides a
compact yet performant baseline through knowledge distillation from BERT; and the full
BERT-based teacher–student framework explicitly transfers soft-label knowledge using a
combined cross-entropy and Kullback–Leibler divergence loss.
Results show that all models effectively separate human and AI-authored text, with
validation accuracies ranging from 97.7% to 98.6%. The distilled student retains over 97%
of the teacher’s accuracy while offering significantly faster inference and lower memory
usage.
