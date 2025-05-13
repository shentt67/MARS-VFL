import sklearn.metrics
import numpy as np


def ptsort(tu):
    return tu[0]


def AUPRC(pts):
    true_labels = [int(x[1]) for x in pts]
    predicted_probs = [x[0] for x in pts]
    return sklearn.metrics.average_precision_score(true_labels, predicted_probs)


def f1_score(truth, pred, average):
    return sklearn.metrics.f1_score(truth, pred, average=average)


def accuracy(truth, pred):
    return sklearn.metrics.accuracy_score(truth.cpu().numpy(), pred.cpu().numpy())


def eval_affect(truths, results, exclude_zero=True):
    if type(results) is np.ndarray:
        test_preds = results
        test_truth = truths
    else:
        test_preds = results.cpu().numpy()
        test_truth = truths.cpu().numpy()

    non_zeros = np.array([i for i, e in enumerate(
        test_truth) if e != 0 or (not exclude_zero)])

    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    return sklearn.metrics.accuracy_score(binary_truth, binary_preds)

def eval_multi_affect(y_true, y_pred):
    test_preds = y_pred.view(-1).cpu().detach().numpy()
    test_truth = y_true.view(-1).cpu().detach().numpy()

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
    test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
    test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)

    mae = np.mean(np.absolute(test_preds - test_truth)).astype(
        np.float64)  # Average L1 distance between preds and truths
    mult_a7 = __multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = __multiclass_acc(test_preds_a5, test_truth_a5)
    mult_a3 = __multiclass_acc(test_preds_a3, test_truth_a3)

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
    if len(non_zeros) == 0:
        eval_results = {
            "mae": round(mae, 4),
            "f1_score": round(0, 4),
            "acc_2": round(0, 4),
            "acc_3": round(mult_a3, 4),
            "acc_5": round(mult_a5, 4),
            "acc_7": round(mult_a7, 4)
        }
    else:
        non_zeros_binary_truth = (test_truth[non_zeros] > 0)
        non_zeros_binary_preds = (test_preds[non_zeros] > 0)

        non_zeros_acc2 = sklearn.metrics.accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
        non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')

        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)
        acc2 = sklearn.metrics.accuracy_score(binary_preds, binary_truth)
        f_score = f1_score(binary_truth, binary_preds, average='weighted')

        eval_results = {
            "mae": round(mae, 4),
            "f1_score": round(non_zeros_f1_score, 4),
            "acc_2": round(non_zeros_acc2, 4),
            "acc_3": round(mult_a3, 4),
            "acc_5": round(mult_a5, 4),
            "acc_7": round(mult_a7, 4)
        }
    return eval_results

def __multiclass_acc(y_pred, y_true):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))