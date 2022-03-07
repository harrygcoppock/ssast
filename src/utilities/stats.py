import numpy as np
from scipy import stats
from sklearn import metrics
import torch
import matplotlib.pyplot as plt

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime
def ciab_metrics(output, target):

    """Calculate statistics: UAR, PR_AUC, ROC_AUC

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
        metrics: dic
    """
    metrics = {}
    uar = recall_score(target, output, average='macro')
    cm = confusion_matrix(target, output)

    #plot_roc_curve(estimator, test_X, test_y, pos_label='Positive')
    fig, prec, rec, pr_auc = PR_AUC(target, output)
    #plt.title(f'{self.modality}_{test_name}')
    #plt.savefig(f'{results_folder}{test_name}_PRcurve.png')
    #plt.close()

    fig, fpr, tpr, roc_auc = ROC_AUC(target, output)
    #plt.plot([0,1], [0, 1], color='red', linestyle='--')
    #plt.title(f'{self.modality}_{test_name}')
    #plt.savefig(f'{results_folder}{test_name}_roc_curve.png')
    #plt.close()

    metrics['precision'] = prec.tolist()
    metrics['recall'] = rec.tolist()
    metrics['fpr'] = fpr.tolist()
    metrics['tpr'] = tpr.tolist()
    metrics['pr_auc'] = pr_auc
    metrics['roc_auc'] = roc_auc
    metrics['uar'] = uar
    metrics['cm'] = cm.tolist()
    return metrics

def PR_AUC(targets, outputs):
    prec, rec, _ = metrics.precision_recall_curve(targets, outputs, pos_label='Positive')
    pr_auc = metrics.average_precision_score(targets, outputs, pos_label='Positive')
    #now plot
    viz = metrics.PrecisionRecallDisplay(recall=rec,
            precision=prec,
            average_precision=pr_auc)
    return viz.plot(), prec, rec, pr_auc

def ROC_AUC(targets, outputs):
    fpr, tpr, _ = metrics.roc_curve(targets, outputs, pos_label='Positive')
    roc_auc = metrics.auc(fpr, tpr)
    #now plot
    viz = metrics.RocCurveDisplay(
            fpr=fpr,
            tpr=tpr,
            roc_auc=roc_auc)
    return viz.plot(), fpr, tpr, roc_auc 
def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))

    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k])

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1000     # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc,
                # note acc is not class-wise, this is just to keep consistent with other metrics
                'acc': acc
                }
        stats.append(dict)

    return stats

