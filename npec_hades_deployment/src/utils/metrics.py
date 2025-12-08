import keras.backend as K


def f1(y_true, y_pred):
    """Calculate F1 score.

    :param y_true: Ground Truth.
    :param y_pred: Prediction.
    :return: F1 score.
    """

    def recall_m(y_true, y_pred):
        """Calculate Recall.

        :param y_true: Ground Truth.
        :param y_pred: Prediction.
        :return: Recall.
        """
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        """Calculate Precision.

        :param y_true: Ground Truth.
        :param y_pred: Prediction.
        :return: Precision.
        """
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives + K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def iou(y_true, y_pred):
    """Calculate Intersection over Union.

    :param y_true: Ground Truth.
    :param y_pred: Prediction.
    :return: IoU score.
    """

    def f(y_true, y_pred):
        """Calculate components of IoU

        :param y_true: Ground Truth.
        :param y_pred: Prediction.
        :return: IoU components.
        """
        threshold = 0.5
        y_pred_binary = K.round(y_pred + 0.5 - threshold)

        intersection = K.sum(K.abs(y_true * y_pred_binary), axis=[1, 2, 3])
        total = K.sum(K.square(y_true), [1, 2, 3]) + K.sum(K.square(y_pred_binary), [1, 2, 3])
        union = total - intersection
        return (intersection + K.epsilon()) / (union + K.epsilon())

    return K.mean(f(y_true, y_pred), axis=-1)
