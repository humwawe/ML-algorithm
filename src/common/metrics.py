def accuracy_score(y_true, y_predict):
    '''y_true, y_predict之间的准确度'''
    assert y_true.shape[0] == y_predict.shape[0], "size must be equal"
    return sum(y_true == y_predict) / len(y_true)
