import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve, auc,
    confusion_matrix, average_precision_score, roc_auc_score
)


def calculate_all_metrics(y_true, y_pred, y_prob=None):
    """Get all metrics in one go"""
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_prob is not None:
        metrics['auc_pr'] = average_precision_score(y_true, y_prob)
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', ax=None):
    cm = confusion_matrix(y_true, y_pred)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    # percentages
    cm_pct = cm.astype('float') / cm.sum() * 100
    
    labels = [[f'{count}\n({pct:.2f}%)' for count, pct in zip(row, pct_row)]
              for row, pct_row in zip(cm, cm_pct)]
    
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax,
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    
    return ax


def plot_pr_roc_curves(y_true, y_prob, model_name='Model', figsize=(12, 5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # PR curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    axes[0].plot(rec, prec, label=f'{model_name} (AUC-PR = {pr_auc:.4f})')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Precision-Recall Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    axes[1].plot(fpr, tpr, label=f'{model_name} (AUC-ROC = {roc_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_multiple_pr_curves(results_dict, y_true, figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, y_prob in results_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, label=f'{name} (AUC-PR = {pr_auc:.4f})')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_multiple_roc_curves(results_dict, y_true, figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, y_prob in results_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC-ROC = {roc_auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def print_metrics_table(metrics_dict):
    print("\n" + "="*70)
    print(f"{'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-PR':>10} {'AUC-ROC':>10}")
    print("="*70)
    
    for name, m in metrics_dict.items():
        print(f"{name:<20} {m.get('precision', 0):>10.4f} {m.get('recall', 0):>10.4f} "
              f"{m.get('f1_score', 0):>10.4f} {m.get('auc_pr', 0):>10.4f} {m.get('auc_roc', 0):>10.4f}")
    
    print("="*70 + "\n")
