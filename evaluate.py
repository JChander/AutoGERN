import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve


def evaluate(predictions_path, labels_path):
    """
    评估预测结果
    
    Args:
        predictions_path: prediction.npy文件路径，shape为(n_samples, 2)
                          第一列是预测为0的logit，第二列是预测为1的logit
        labels_path: labels.npy文件路径，shape为(n_samples,)，每个元素是对应的标签
    """
    # 加载数据
    predictions = np.load(predictions_path)
    labels = np.load(labels_path)
    
    # 验证数据形状
    assert predictions.ndim == 2, f"predictions应该是2维数组，当前是{predictions.ndim}维"
    assert predictions.shape[1] == 2, f"predictions应该有2列，当前有{predictions.shape[1]}列"
    assert labels.ndim == 1, f"labels应该是1维数组，当前是{labels.ndim}维"
    assert predictions.shape[0] == labels.shape[0], \
        f"predictions和labels的样本数不一致: {predictions.shape[0]} vs {labels.shape[0]}"
    
    # 计算Accuracy
    # 从logits中获取预测类别（argmax）
    predicted_classes = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predicted_classes)
    
    # 计算AUROC
    # 使用softmax将logits转换为概率，取第二列（类别1的概率）
    logits = predictions
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # 数值稳定性
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    prob_class1 = probs[:, 1]  # 类别1的概率
    
    # 对于二分类，使用正类（类别1）的概率计算AUROC
    try:
        auroc = roc_auc_score(labels, prob_class1)
    except ValueError as e:
        # 如果只有一个类别，AUROC无法计算
        print(f"Warning: {e}")
        auroc = np.nan
    
    # 计算AUPRC (Average Precision)
    try:
        auprc = average_precision_score(labels, prob_class1)
    except ValueError as e:
        print(f"Warning: {e}")
        auprc = np.nan
    
    # 打印结果
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Total samples: {len(labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print("=" * 50)
    
    return {
        'accuracy': accuracy,
        'auroc': auroc,
        'auprc': auprc
    }

def visualize_predictions(predictions_path, labels_path):
    """
    可视化预测结果
    """
    predictions = np.load(predictions_path)
    labels = np.load(labels_path)
    
    # plot ROC curve
    fpr, tpr, thresholds = roc_curve(labels, predictions[:, 1])
    plt.plot(fpr, tpr, label='ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.pdf')
    plt.close()
    # plot PR curve
    precision, recall, thresholds = precision_recall_curve(labels, predictions[:, 1])
    plt.plot(recall, precision, label='PR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.legend()
    plt.savefig('pr_curve.pdf')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate predictions for binary classification')
    parser.add_argument('--predictions', '-p', type=str, required=True,
                        help='Path to prediction.npy file (shape: n_samples, 2)')
    parser.add_argument('--labels', '-l', type=str, required=True,
                        help='Path to labels.npy file (shape: n_samples,)')
    
    args = parser.parse_args()
    
    evaluate(args.predictions, args.labels)
    # visualize_predictions(args.predictions, args.labels)


if __name__ == '__main__':
    main()













