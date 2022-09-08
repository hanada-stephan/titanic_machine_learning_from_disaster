def run_model(model, X_train, X_test, y_train):
    """Run sklearn models
    
    Args:
        model (sklearn model): model to run
        X_train (Pandas series, nparray): Training data set
        X_test (Pandas series, nparray): Test data set
        y_train (Pandas series, nparray): Target of training data set

    Returns:
        array with predictions
    """ 
    model_inst = model.fit(X_train, y_train)
    y_pred = model_inst.predict(X_test)
    return y_pred