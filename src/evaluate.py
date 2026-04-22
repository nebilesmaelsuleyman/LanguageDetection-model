from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(trainer, tokenized_ds, label_encoder):
    preds = trainer.predict(tokenized_ds["validation"])
    pred_labels = np.argmax(preds.predictions, axis=-1)
    true_labels = tokenized_ds["validation"]["labels"]

    print("--- Classification Report ---")
    print(classification_report(true_labels, pred_labels, target_names=label_encoder.classes_))

    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Save instead of show to avoid blocking in scripts
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix as confusion_matrix.png")
