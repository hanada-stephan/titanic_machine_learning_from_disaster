from sklearn.metrics import accuracy_score,\
                            precision_score,\
                            recall_score

def validation_scores(y_test, y_pred):
    """Print validations scores
    
    Args:
        y_test (Pandas series, nparray): Target of test data set
        y_pred (nparray): Predicted target

    Returns:
        Print of all three validation scores
    """ 

    print("Accuracy score: ", accuracy_score(y_test, y_pred))
    print("Precision score: ",precision_score(y_test, y_pred))
    print("Recall score: ", recall_score(y_test, y_pred))