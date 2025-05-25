import numpy as np

def mean_average_precision(y_true, y_pred, k=None):
    """
    Calculate Mean Average Precision (MAP) for a set of predictions.
    
    MAP is the mean of Average Precision scores for a set of queries/users.
    It is commonly used to evaluate recommendation systems and ranking tasks.
    
    Args:
        y_true: List of lists where each sublist contains the true relevant items 
                for a particular user/query.
        y_pred: List of lists where each sublist contains the predicted items 
                in descending order of relevance for a particular user/query.
        k: Optional, compute MAP@k which considers only the top k predictions.
                If None, uses all predictions.
    
    Returns:
        float: The MAP score
    
    Example:
        >>> y_true = [[1, 3], [2, 4, 7]]
        >>> y_pred = [[1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7]]
        >>> mean_average_precision(y_true, y_pred)
        0.75
        >>> mean_average_precision(y_true, y_pred, k=2)
        0.5
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        raise ValueError("Input lists must be non-empty and of the same length")
    
    aps = []
    
    for i, (true_items, pred_items) in enumerate(zip(y_true, y_pred)):
        if not true_items:
            continue
            
        if k is not None:
            pred_items = pred_items[:k]
            
        # Calculate average precision for this user/query
        ap = average_precision(true_items, pred_items)
        aps.append(ap)
    
    if not aps:
        return 0.0
        
    # Mean Average Precision is the mean of all Average Precision scores
    return np.mean(aps)

def average_precision(true_items, pred_items):
    """
    Calculate Average Precision for a single list of predictions.
    
    AP summarizes the precision-recall curve as the weighted mean of precisions 
    achieved at each threshold, with the increase in recall from the previous 
    threshold used as the weight.
    
    Args:
        true_items: List of true relevant items.
        pred_items: List of predicted items in descending order of relevance.
        
    Returns:
        float: The Average Precision score
    """
    if not true_items or not pred_items:
        return 0.0
        
    hits = 0
    sum_precisions = 0.0
    
    for i, item in enumerate(pred_items):
        if item in true_items:
            hits += 1
            # Precision at rank i+1
            precision = hits / (i + 1)
            sum_precisions += precision
    
    # Normalize by the total number of relevant items
    return sum_precisions / len(true_items) if true_items else 0.0
