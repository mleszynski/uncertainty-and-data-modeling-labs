"""
Random Forest Lab

Marcelo Leszynski
Section 001
5 October 2021
"""
import graphviz
import os
from uuid import uuid4
from sklearn.ensemble import RandomForestClassifier
from time import time
import numpy as np

# Problem 1
class Question:
    """Questions to use in construction and display of Decision Trees.
    Attributes:
        column (int): which column of the data this question asks
        value (int/float): value the question asks about
        features (str): name of the feature asked about
    Methods:
        match: returns boolean of if a given sample answered T/F"""
    
    def __init__(self, column, value, feature_names):
        self.column = column
        self.value = value
        self.features = feature_names[self.column]
    
    def match(self,sample):
        """Returns T/F depending on how the sample answers the question
        Parameters:
            sample ((n,), ndarray): New sample to classify
        Returns:
            (bool): How the sample compares to the question"""
        return sample[self.column] >= self.value
        
    def __repr__(self):
        return "Is %s >= %s?" % (self.features, str(self.value))
    
def partition(data,question):
    """Splits the data into left (true) and right (false)
    Parameters:
        data ((m,n), ndarray): data to partition
        question (Question): question to split on
    Returns:
        left ((j,n), ndarray): Portion of the data matching the question
        right ((m-j, n), ndarray): Portion of the data NOT matching the question
    """
    l = []
    r = []
    
    # partition across question ################################################
    for i in range(data.shape[0]):
        if question.match(data[i,:]):
            l.append(data[i,:])
        else:
            r.append(data[i,:])
            
    # handle edge cases ########################################################
    if len(l) >0:
        l = np.stack(l)
    else:
         l = None
    if len(r) > 0:
        r = np.stack(r)
    else:
        r = None
    return l,r
    
# Problem 2    
def gini(data):
    """Return the Gini impurity of given array of data.
    Parameters:
        data (ndarray): data to examine
    Returns:
        (float): Gini impurity of the data"""
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=0)

    # calculate gini ###########################################################
    val,counts = np.unique(data[:,-1],return_counts=True)
    n = len(data[:,1])
    gd = 1 - np.sum((counts/n)**2)
    return gd
    
    

def info_gain(left,right,G):
    """Return the info gain of a partition of data.
    Parameters:
        left (ndarray): left split of data
        right (ndarray): right split of data
        G (float): Gini impurity of unsplit data
    Returns:
        (float): info gain of the data"""
        
    # handle edge cases ########################################################
    if len(left.shape) == 1:
        left = np.expand_dims(left, axis=0)
    if len(right.shape) == 1:
        right = np.expamd_dims(right,axis=0)
        
    # calculate using gini #####################################################
    l = left.shape[0]
    r = right.shape[0]
    gl = gini(left)
    gr = gini(right)
    
    return G - (l*gl)/(l+r) - (r*gr)/(l+r)
    
    
# Problem 3, Problem 7
def find_best_split(data, feature_names, min_samples_leaf=5, random_subset=False):
    """Find the optimal split
    Parameters:
        data (ndarray): Data in question
        feature_names (list of strings): Labels for each column of data
        min_samples_leaf (int): minimum number of samples per leaf
        random_subset (bool): for Problem 7
    Returns:
        (float): Best info gain
        (Question): Best question"""
        
    # handle the random subset option ##########################################
    if random_subset:
        size = int(np.sqrt(len(feature_names)-1))
        i = list(range(len(feature_names)-1))
        np.random.shuffle(i)
        ind = i[:size]
    # handle the other case ####################################################
    else:
        ind = range(data.shape[1]-1)

    # initialize variables #####################################################
    max_ent = 0
    max_q = None
    G = gini(data)
    
    # calculate best info gain and question ####################################
    for i in ind:
        un = np.unique(data[:,i])
        
        for j in un:
            q = Question(i,j,feature_names)
            left,right = partition(data, q)
            if left is not None and right is not None:
                if len(left) >= min_samples_leaf and len(right)>= min_samples_leaf:
                    ent = info_gain(left, right, G)
                    if ent > max_ent:
                        max_ent = ent
                        max_q = q
                        
    return (max_ent,max_q)
    

# Problem 4
class Leaf:
    """Tree leaf node
    Attribute:
        prediction (dict): Dictionary of labels at the leaf"""
    def __init__(self,data):
        unique, counts = np.unique(data[:,-1], return_counts=True)
        self.prediction = dict(zip(unique,counts))
    
    def isleaf(self):
        return True
        

class Decision_Node:
    """Tree node with a question
    Attributes:
        question (Question): Question associated with node
        left (Decision_Node or Leaf): child branch
        right (Decision_Node or Leaf): child branch"""
    def __init__(self, question, right_branch, left_branch):
        self.question = question
        self.left = left_branch
        self.right = right_branch

    def isleaf(self):
        return False
        

## Code to draw a tree
def draw_node(graph, my_tree):
    """Helper function for drawTree"""
    node_id = uuid4().hex
    #If it's a leaf, draw an oval and label with the prediction
    if isinstance(my_tree, Leaf):
        graph.node(node_id, shape="oval", label="%s" % my_tree.prediction)
        return node_id
    else: #If it's not a leaf, make a question box
        graph.node(node_id, shape="box", label="%s" % my_tree.question)
        left_id = draw_node(graph, my_tree.left)
        graph.edge(node_id, left_id, label="T")
        right_id = draw_node(graph, my_tree.right)    
        graph.edge(node_id, right_id, label="F")
        return node_id

def draw_tree(my_tree):
    """Draws a tree"""
    #Remove the files if they already exist
    for file in ['Digraph.gv','Digraph.gv.pdf']:
        if os.path.exists(file):
            os.remove(file)
    graph = graphviz.Digraph(comment="Decision Tree")
    draw_node(graph, my_tree)
    graph.render(view=True) #This saves Digraph.gv and Digraph.gv.pdf

# Prolem 5
def build_tree(data, feature_names, min_samples_leaf=5, max_depth=4, current_depth=0, random_subset=False):
    """Build a classification tree using the classes Decision_Node and Leaf
    Parameters:
        data (ndarray)
        feature_names(list or array)
        min_samples_leaf (int): minimum allowed number of samples per leaf
        max_depth (int): maximum allowed depth
        current_depth (int): depth counter
        random_subset (bool): whether or not to train on a random subset of features
    Returns:
        Decision_Node (or Leaf)"""
        
    # handle the base case #####################################################
    if current_depth == max_depth or data.shape[0] <  2* min_samples_leaf:
        return Leaf(data)
    else:
        # find a split #########################################################
        ent,q = find_best_split(data, feature_names,min_samples_leaf,random_subset)
        
        # handle the base case #################################################
        if ent == 0:
             return Leaf(data)
        # handle the recursive case ############################################
        else:
            left,right = partition(data, q)
            return Decision_Node(q,
                                 build_tree(right,feature_names,min_samples_leaf,max_depth,current_depth+1,random_subset), 
                                 build_tree(left,feature_names,min_samples_leaf,max_depth,current_depth+1,random_subset))
             
# Problem 6
def predict_tree(sample, my_tree):
    """Predict the label for a sample given a pre-made decision tree
    Parameters:
        sample (ndarray): a single sample
        my_tree (Decision_Node or Leaf): a decision tree
    Returns:
        Label to be assigned to new sample"""
     
    # handle the base case #####################################################
    if my_tree.isleaf():
        
        # make a prediction ####################################################
        k,v = list(my_tree.prediction.keys()),list(my_tree.prediction.values())
        if len(k)> 1:
            return k[np.argmax(v)]
        else: 
            return k
        
    # handle the recursive case ################################################    
    else:
        q = my_tree.question
        if q.match(sample):
            return predict_tree(sample,my_tree.left)
        else:
            return predict_tree(sample,my_tree.right)
    
def analyze_tree(dataset,my_tree):
    """Test how accurately a tree classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        tree (Decision_Node or Leaf): a decision tree
        
    Returns:
        (float): Proportion of dataset classified correctly"""
    num_corr = [1 for i in range(dataset.shape[0]) if predict_tree(dataset[i,:],my_tree) == dataset[i,-1]]
    return len(num_corr)/dataset.shape[0]


# Problem 7
def predict_forest(sample, forest):
    """Predict the label for a new sample, given a random forest
    Parameters:
        sample (ndarray): a single sample
        forest (list): a list of decision trees
    Returns:
        Label to be assigned to new sample"""
    predictions = [predict_tree(sample, i) for i in forest]
    return np.bincount(predictions).argmax()

def analyze_forest(dataset,forest):
    """Test how accurately a forest classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        forest (list): list of decision trees
    Returns:
        (float): Proportion of dataset classified correctly"""
    num_corr = [1 for i in range(dataset.shape[0]) if predict_forest(dataset[i,:],forest) == dataset[i,-1]]
    return len(num_corr)/dataset.shape[0]
    

# Problem 8
def prob8():
    """Use the file parkinsons.csv to analyze a 5 tree forest.
    
    Create a forest with 5 trees and train on 100 random samples from the dataset.
    Use 30 random samples to test using analyze_forest() and SkLearn's 
    RandomForestClassifier.
    
    Create a 5 tree forest using 80% of the dataset and analzye using 
    RandomForestClassifier.
    
    Return three tuples, one for each test.
    
    Each tuple should include the accuracy and time to run: (accuracy, running time) 
    """
    # read data and initialize variables #######################################
    park = np.loadtxt('parkinsons.csv', delimiter=',')[:,1:]
    features = np.loadtxt('parkinsons_features.csv', delimiter=',', dtype=str,comments=None)
    shuffled = np.random.permutation(park)
    train = shuffled[:100]
    test = shuffled[100:130]
    t_1 = time()
    
    # create a forrest #########################################################
    forest = [ build_tree(train, features,min_samples_leaf=15)for i in range(5)]
    acc = analyze_forest(test, forest)
    ans_1 = (acc,time()-t_1)
    
    # now using sklearn ########################################################
    t_1 = time()
    forest = RandomForestClassifier(n_estimators=5, max_depth=4, min_samples_leaf=15)
    forest.fit(train[:,:-1], train[:,-1])
    acc = forest.score(test[:,:-1], test[:,-1])
    ans_2 = (acc,time()-t_1)
    
    # get random sample of 80/20 split #########################################
    _80_perc = int(park.shape[0] * .8)
    train = shuffled[:_80_perc]
    test = shuffled[_80_perc:]
    
    # test the whole dataset ###################################################
    t_1 = time()
    forest = RandomForestClassifier()
    forest.fit(train[:,:-1], train[:,-1])
    acc = forest.score(test[:,:-1], test[:,-1])
    ans_3 = (acc,time()-t_1)
    
    return ans_1,ans_2,ans_3