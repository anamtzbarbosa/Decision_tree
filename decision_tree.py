import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt
import random

"""
Steps to the decision tree:

1. Choose best characteritic and division method (entropy and information gain).
  1.1 For each node  
    1.2 for each characteristic calculate entropy and information gain.
      1.3 Choose best characteristic and threshold value that minimizes entropy and maximizes info gain.
2. From the best characteristic and threshold, divide set of data in 2 sets (yes/no): 
3. For each new branch (yes/no) or (left/right), repeat 1 and 2 unitl a stopping criteria is met.
  Max depth, branch purity, etc.
4. Label each leaf depending to the mayoritarian class of the subset.
5. Predictions: To make predictions with your trained tree, you walk the tree from root to leaf 
    following decisions based on the characteristics of the instance you want to predict. 
    The class label assigned to the sheet you end up on will be your prediction.

"""

def impurity_entropy(columna):

  """
  - The entropy is a concept that meassures the impurity of the sample.
  - It is used to select the best characteristic to divide. 
  - Goes from 0 to 1, and we choose the clossest to 0 (pure).
  """
  # Function implementation here
  counts = columna.value_counts().values
  prob = (counts)/sum(counts)
  entropy = -np.sum(np.log2(prob)*prob)
  return entropy

def E_split(branch_2_1,branch_2_2):

  E_left = impurity_entropy(branch_2_1)
  E_right = impurity_entropy(branch_2_2)
  E_split = E_left*(len(branch_2_1))/(len(branch_2_1)+len(branch_2_2)) + E_right*(len(branch_2_2))/(len(branch_2_1)+len(branch_2_2))
  return E_split


def info_gain(y,s_condicion): 
  """
  - The attribute with greatest information gain will classify the best the training set.

  1. From divison condition, create the 2 subsets
  2. Obtain information gain for that division

    - y : parent node before divison
    - s_condicion : condition for splitting subsets 7 mask. 

  """
  E_d = impurity_entropy(y) #E parent
 
  s_1_subset= y[s_condicion] 
  s_2_subset= y[-s_condicion]

  s_1_count = len(s_1_subset) #calcula cant de una división
  s_2_count = len(s_2_subset) #cant de otra división

  if s_1_count == 0 and s_2_count == 0: # if the branch is empty / no data fulfills condition
    return 0
 
  if y.dtypes != 'O' : # if categorical classification
    return info_gain_classification(y, s_1_subset,s_2_subset)
  else: # if regression
    return info_gain_regression(y, s_1_subset,s_2_subset)

def info_gain_classification(y, s_1_subset,s_2_subset):
  """
  - y -> Parent node before split
  - s_1 subset and  s_2 subset -> subsets from split
  
  Calculates the informatin gain for the split given as argument

  """
  s_1_count = len(s_1_subset) #number elements in each subset
  s_2_count = len(s_2_subset)  
  ponderado_s1 = s_1_count / (s_1_count + s_2_count) #probability of each subset
  ponderado_s2 = s_2_count / (s_1_count + s_2_count)
  #uses impurity to calculate information gain in classification
  gain = impurity_entropy(y) - (ponderado_s1 * impurity_entropy(s_1_subset) + ponderado_s2 * impurity_entropy(s_2_subset)) 

  return gain

def info_gain_regression(y, s_1_subset,s_2_subset): 
  s_1_count = len(s_1_subset) #number elements in each subset
  s_2_count = len(s_2_subset) 
  ponderado_s1 = s_1_count / (s_1_count + s_2_count)
  ponderado_s2 = s_2_count / (s_1_count + s_2_count)
  # uses variance to calculate information gain in regression
  gain = y.var() - (ponderado_s1 * s_1_subset.var() + ponderado_s2 * s_2_subset.var()) 

  return gain


def all_split_values(predictor,target):
  """
  From the functions established before to calculate information gain and entropy, 
  it will check for different splits, calculate infromation gain and decie what is the best
  split.

    - predictor -> column that is going to be splitted into subsets.
    - target -> target column for classification

  """
  all_info_gains = []
  split_value = []

  #For regression
  if predictor.dtypes != 'O': 
    options = predictor.sort_values().unique() 
    #find all different possible options for split
 
    for i in options:
      # condition for  splitting /mask : predictor element < i (iters for all possible values of i)
      info_gains = info_gain(target, predictor < i)
      # Appends the corresponding splitting value and info gain
      all_info_gains.append(info_gains)
      split_value.append(i)

  #For classification
  else: 
    options = predictor.unique() #find all different possible options for split
 
    for i in options: 
      # mask ej data['Gender'] == 'Male'
      # The result is an array of True/False
      info_gains = info_gain(target, predictor == i)
      all_info_gains.append(info_gains)
      split_value.append(i)

  return best_split_val(all_info_gains , split_value)

def best_split_val(all_info_gains , split_value):

  """
  - all_info_gains: array of all info gains
  - split_value: array of all splitting values

  The function chooses the best splitting value by minimizing the decision tree's loss function
  In other words, maximizing the information gain.

  """
  if len(all_info_gains) != 0:
    # returns splitting value and corresponding info gain
    return (split_value[all_info_gains.index(max(all_info_gains))], max(all_info_gains))
  else:
    return False

def best_split_fromall(data , y):
  """
  Selects the variable with the best information gain from 
  all the possible columns to choose.

    - data : complete df without target column
    - y : target column
  
  """
  column_names = data.columns.tolist()
  all_info_gains =[]
  split_val = []
  for i in range(len(column_names)):
    all_info_gains.append(all_split_values(data[column_names[i]] , y )[1])
    split_val.append(all_split_values(data[column_names[i]],y )[0])
    # returns column name and splitting value
  return (column_names[all_info_gains.index(max(all_info_gains))], split_val[all_info_gains.index(max(all_info_gains))], max(all_info_gains))

def split(data, y):
  """
  Creates 2 arrays with the corresponding elements according to the 
  splitting conditin and value from best_split_fromall.
    - data : complete df without target column
    - y : target column
  
  """ 
  best_split_= best_split_fromall(data, y)
  if best_split_ == False:
    return (False)
  else:
    if isinstance(best_split_[1], str): # For classification
      data_1 = data[data[best_split_[0]].isin([best_split_[1]])]
      data_2 = data[(data[best_split_[0]].isin([best_split_[1]])) == False]
    
    #For regression
    else:
      data_1 = data[data[best_split_[0]] < best_split_[1]]
      data_2 = data[data[best_split_[0]] >= best_split_[1]]
  return (data_1,data_2)

def make_prediction(y):
  """
  Whenever a branch is empty and there is need to continue the iterations
  it makes a predicition from the most repeated value (mode) or mean.
    - y : target column

  """
  if isinstance(y,str):
    pred = y.mode().iloc[0]
  else:
    pred = y.mean()
    
    return pred #moda


def train_tree(data, y, counter):
    """
    Train recursively

    1. Choose best characteritic and division method (entropy and information gain).
    2. From the best characteristic and threshold, divide set of data in 2 sets (yes/no): 
    3. For each new branch (yes/no) or (left/right), repeat 1 and 2. 
    
      - data : complete df without target column
      - y : target column
    """
    
    if isinstance(data, pd.DataFrame) and len(data) > 0:  # Check if data is a valid DataFrame
        best_split_var, best_split_value, _ = best_split_fromall(data, y)

        if best_split_var is None:  # No valid split found
            pred = make_prediction(y)
            return pred

        # Categorical split
        if isinstance(best_split_value, str):  
            mask = data[best_split_var] == best_split_value  # array containing only T / F depending on condition
            
        else:  # Numerical split
            mask = data[best_split_var] < best_split_value # array containing only T / F depending on condition

        data_left = data[mask] # array containing only T values when condition met
        data_right = data[~mask] # array containing only F, theht indicate those ntries from the other branch
        y_left = y[mask] # array containing only corresponding target values for T entries
        y_right = y[~mask] # array containing only corresponding target values for F entries

        if len(data_left) == 0 or len(data_right) == 0:  # SEMpty branch
            pred = make_prediction(y)
            return pred

        
        #Format way of displaying the three execution for regression and classification
        split_type = "in" if isinstance(best_split_value, str) else "<="
        step =  f"{best_split_var} {split_type} {best_split_value}"
        subtree = {step: []}

        #Repeat recursively 
        yes = train_tree(data_left, y_left, counter)
        no = train_tree(data_right, y_right, counter)

        #Append the new branches to the according parent nodes
        subtree[step].append(yes)
        subtree[step].append(no)
        return subtree
    else:
        return None


def predict_example(tree, example):
    """
    Recursively label each leaf depending to the mayoritarian class of the subset.

    """
    for condition, sub_tree in tree.items():
        variable, comparison, value = condition.split()
      # Condición categórica
        if comparison == "<=": # Yes branch
            if example[variable] <= float(value):
                if isinstance(sub_tree[0], dict):  # if subtree has more nodes
                    return predict_example(sub_tree[0], example) # recusrively keeps going a leve down in tree and makes prediction
                else:  # if leaf node
                    return sub_tree[0] # return predited class
            else: #lado no
                if isinstance(sub_tree[1], dict):  
                    return predict_example(sub_tree[1], example)
                else: 
                    return sub_tree[1]

        # Condición numérica

        else: 
            if example[variable] == value:
                if isinstance(sub_tree[0], dict): 
                    return predict_example(sub_tree[0], example)
                else: 
                    return sub_tree[0]
            else:
                if isinstance(sub_tree[1], dict): 
                    return predict_example(sub_tree[1], example)
                else: 
                    return sub_tree[1]
                


def predict_examples(tree, test_data):
    """
    Uses fucntion predict_example to keep predicting for all rows in the dataframe to be labelled
      - tree : decsision tree calculates
      - test_data : df to pe labelled
    """
    predictions = [] 
    for index, row in test_data.iterrows():
        prediction = predict_example(tree, row)
        predictions.append(prediction)

    return predictions

###############################################################

#Upload Dataset

data = pd.read_csv('Person_Gender_Height_Weight_Index.csv') #Change to your own path
data['obese'] = (data.Index >= 4).astype('int')
data.drop('Index', axis = 1, inplace = True)

results_df = pd.DataFrame(columns=['Random State', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'])
random_state_array = []
test_labels =[]
# Perform multiple tests with different random states
for i in range(20):
    # Split the data
    name = 'Test '+str(i)
    test_labels.append(name)
    random_state = random.randint(1, 100)
    random_state_array.append(random_state)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=random_state)

    # Train the decision tree
    decision_tree = train_tree(train_data.drop(columns=['obese']), train_data['obese'], 0)

    # Make predictions
    predictions = predict_examples(decision_tree, test_data.drop(columns=['obese']))

    # Calculate classification metrics
    accuracy = sklearn.metrics.accuracy_score(test_data['obese'], predictions)
    precision = sklearn.metrics.precision_score(test_data['obese'], predictions)
    recall = sklearn.metrics.recall_score(test_data['obese'], predictions)
    f1_score = sklearn.metrics.f1_score(test_data['obese'], predictions)
    roc_auc = sklearn.metrics.roc_auc_score(test_data['obese'], predictions)

    results_df = results_df.append({'Random State': random_state,
                                    'Accuracy': accuracy,
                                    'Precision': precision,
                                    'Recall': recall,
                                    'F1-Score': f1_score,
                                    'ROC AUC': roc_auc}, ignore_index=True)

# Display the results 
print(results_df)
#test_labels = ['Test 1', 'Test 2', 'Test 3', 'Test 4']
plt.figure(figsize=(8, 6))
plt.plot(test_labels, results_df['Accuracy'], marker='o', linestyle='-', color='b', label='Accuracy')
plt.xlabel('Tests')
plt.ylabel('Accuracy')
plt.title('Algorithm Generalization Across Tests')
plt.legend()
plt.grid(True)

# Display or save the graph
plt.show()
#We can appreciate that the algorithm mantains high metrics, therefore is good at generalizing the data.