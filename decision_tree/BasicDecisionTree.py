import numpy as np

class Node:

    """
    Represents a node in the decision tree.

    Attributes:
        feature (int): Index of the feature used for splitting. Default to None.
        threshold (float): Threshold value for the split. Default to None.
        left (Node): Left child node. Default to None.
        right (Node): Right child node. Default to None.
        value: The value predicted by this node. For classification, it's the class. For regression, it's a numerical value. Defaults to None.
        impurity (float): Impurity measure of the node.
        is_leaf (bool): Flag indicating if the node is a leaf node.
        n_samples (int): Number of samples in the node. Default to None
    """
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, impurity=0.0, is_leaf=False, n_samples=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.impurity = impurity
        self.is_leaf = is_leaf
        self.n_samples = n_samples


class DecisionTree:
    
    """
    Decision Tree for classification or regression.

    Attributes:
        max_depth (int): The maximum depth of the tree.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        min_impurity_decrease (float): A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
        task (str): Task type ('classification' or 'regression').
        impurity (str): The impurity measure to use ('gini', 'entropy' for classification and 'mse' for regression).
        alpha (float): Complexity parameter used for pruning.
        root (Node): The root node of the tree.
    """
        
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 min_impurity_decrease=0.0, task='classification', impurity='gini', alpha=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.task = task
        self.impurity = impurity
        self.root = None
        self.alpha = alpha
        
    def _calculate_gini(self, y):
        """
        Calculate the Gini impurity for a set of labels.
        
        Parameters:
        y (array-like): Array of labels in a node.
        
        Returns:
        float: Gini impurity
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities**2)

    def _calculate_entropy(self, y):
        """
        Calculate the Entropy impurity for a set of labels.
        
        Parameters:
        y (array-like): Array of labels in node.
        
        Returns:
        float: Entropy impurity
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = np.clip(counts / counts.sum(), 1e-10, 1)
        return -np.sum(probabilities * np.log2(probabilities))

    def _calculate_mse(self, y):
        """
        Calculate the Mean Square Error for a set of continuous labels.
        
        Parameters:
        y (array-like): Array of continuos labels in a node.
        
        Returns:
        float: Mean Squarred Error.
        """
        return np.mean((y - np.mean(y)) ** 2)

    def _calculate_impurity(self, y):
        """
        Calculate the impurity of a node based on the specified impurity criterion.
        
        Parameters:
        y (array-like): Array of labels or continuous values in a node.
        
        Returns:
        float: Impurity of the node.
        
        """
        if self.task == 'classification':
            if self.impurity == 'gini':
                return self._calculate_gini(y)
            elif self.impurity == 'entropy':
                return self._calculate_entropy(y)
            else:
                raise ValueError(f"Unknown impurity criterion: {self.impurity}")
        elif self.task == 'regression':
            if self.impurity == 'mse':
                return self._calculate_mse(y)
            else:
                raise ValueError(f"Unknown task: {self.task}")
    
    # Split the data into left and right based on the feature and threshold
    def _split_data(self, X, y, feature_index, threshold):
        """
        Split the data into left and right branches based on a feature and threshold.
        
        parameters:
        X (array-like): Input features.
        y (array-like): Output features.
        feature_index (int): Index of the feature to used for splitting.
        threshold (float): Threshold value to use for splitting.
        
        Returns:
        Tuple: Split dataset (X_left, y_left, X_right, y_right).
        """
        
        left_mask = X[:, feature_index] < threshold
        right_mask = ~left_mask
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        return X_left, y_left, X_right, y_right
    
    # Find the best split for the given data
    def _best_split(self, X, y):
        """
        Find the best feature and threshold for splitting the data to maximize impurity reduction.
        
        Parameters:
        X (array-like): Input features.
        y (array-like): Output labels or values.
        
        Returns:
        Tuple: Best feature index, threshold, and gain from the split.
        """
        
        n_samples = len(y)
        best_feature, best_threshold, best_gain = None, None, float('-inf')
        current_impurity = self._calculate_impurity(y)
        for feature_idx in range(self.n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self._split_data(X, y, feature_idx, threshold)
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                left_impurity = self._calculate_impurity(y_left)
                right_impurity = self._calculate_impurity(y_right)
                weighted_impurity = (len(y_left) * left_impurity + len(y_right) * right_impurity) / n_samples
                impurity_gain = current_impurity - weighted_impurity
                if impurity_gain > best_gain and impurity_gain >= self.min_impurity_decrease:
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_gain = impurity_gain
        return best_feature, best_threshold

    # Calculate the value to be predicted by a leaf node
    def _calculate_leaf_value(self, y):
        """
        Calculate the value to be assigned to a leaf node.
        
        Parameters:
        y (array-like): Labels or values in the leaf node.
        
        Returns:
        Class label or mean value, depending on the task.
        """
        
        if self.task == 'classification':
            unique_classes, counts = np.unique(y, return_counts=True)
            index = counts.argmax()
            return unique_classes[index]
        elif self.task == 'regression':
            return np.mean(y)
    
     # Recursively build the decision tree
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        
        Parameters:
        X (array-like): Input features.
        y (array-like): Output labels or values.
        depth (int): Current depth of the tree.
        
        Returns:
        Node: The built tree node.
        """
       
        n_samples = len(y)
        node_impurity = self._calculate_impurity(y)
        node_value = self._calculate_leaf_value(y)
        node = Node(value=node_value, impurity=node_impurity, is_leaf=True, n_samples=n_samples)
        
        if (self.max_depth is None or depth < self.max_depth) and len(y) >= self.min_samples_split and len(np.unique(y)) > 1:
            best_feature, best_threshold = self._best_split(X, y)
            if best_feature is not None:
                X_left, y_left, X_right, y_right = self._split_data(X, y, best_feature, best_threshold)
                left_node = self._build_tree(X_left, y_left, depth + 1)
                right_node = self._build_tree(X_right, y_right, depth + 1)

                node.feature = best_feature
                node.threshold = best_threshold
                node.left = left_node
                node.right = right_node
                node.is_leaf = False

        return node
        
    # Fit the decision tree model on the training data
    def fit(self, X, y):
        """
        Fit the decision tree model to the trainning data/
        
        Parameters:
        X (array-like): Training input features.
        y (array-like): Trainin output labels or values.
        """
        
        X = np.array(X).copy()
        y = np.array(y).copy()
        self.n_samples, self.n_features = X.shape
        self.root = self._build_tree(X, y)
        self._prune_tree()


    # Predicted the value for a single sample
    def _predict_one(self, x):
        """
        Predict the value for single instance.
        
        Parameters:
        
        X (array-like): A single input feature vector.
        
        Returns:
        Predicted class label or value.
        """
        
        node = self.root
        while node is not None and not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value if node is not None else None

    # Predicted the value for multiple samples
    def predict(self, X):
        """
        Predict the value for multiple instances.
        
        Parameters:
        X (array-like): Multiple input feature vectors.
        
        Returns:
        array-like: Predicted class labels or values.
        """
        X = np.array(X)
        return np.array([self._predict_one(sample) for sample in X])

    # Prune the tree using the complexity parameter of alpha
    def _prune_tree(self, node=None):
        """
        Prune the tree using the complexity parameter alpha
        
        Parameters:
        node (Node, optional): Current node to start pruning data
        
        Returns:
        Tuple: Pruned node, its impurity, and number of leaves.
        """
        if node is None:
            node = self.root
                    
        if node.is_leaf:
            return node, node.impurity, 1

        # Recursive calls with alpha parameter
        node.left, left_impurity, left_leaves = self._prune_tree(node.left) if node.left else (None, 0, 0)
        node.right, right_impurity, right_leaves = self._prune_tree(node.right) if node.right else (None, 0, 0)

        total_impurity = left_impurity + right_impurity
        total_leaves = left_leaves + right_leaves

        if node.impurity + self.alpha * total_leaves <= total_impurity:
            node.left, node.right = None, None
            node.is_leaf = True
            return node, node.impurity, 1
        
        return node, total_impurity, total_leaves

    # Print the structure of the tree
    def print_tree(self, node=None, depth=0, feature_names=None):
        """
        Print the structure of the tree.
        
        Parameters:
        node (Node, optional): Current node to print.
        depth(int): Current depth in the tree (used for indentation).
        feature_names (list, optional): List of feature names for better readability.
        """
        if node is None:
            node = self.root

        if feature_names is not None and node.feature is not None and node.feature < len(feature_names):
            feature_name = feature_names[node.feature]
        else:
            feature_name = f"Feature[{node.feature}]"

        if node.is_leaf:
            print(f'{"|   " * depth}--> Predicted value: {node.value}')
        else:
            print(f'{"|   " * depth}Is {feature_name} <= {node.threshold}?')
            self.print_tree(node.left, depth + 1, feature_names)
            print(f'{"|   " * depth}Is {feature_name} > {node.threshold}?')
            self.print_tree(node.right, depth + 1, feature_names)
