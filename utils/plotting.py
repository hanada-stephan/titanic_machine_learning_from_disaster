from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(y_test, y_pred):
    """Plot a confusion matrix
    
    Args:
        y_test (pandas series, nparray): Target test set
        y_pred (pandas series, nparray): Predicted targets

    Returns:
        Figure containing the confusion matrix
    """    

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    