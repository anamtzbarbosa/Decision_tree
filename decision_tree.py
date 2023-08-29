import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn


def impurity_entropy(columna):
  count_as_array_np = columna.value_counts().values
  prob = (count_as_array_np)/sum(count_as_array_np)
  entropy = -np.sum(np.log2(prob)*prob)
  return entropy
def E_split(branch_2_1,branch_2_2):
  E_left = impurity_entropy(branch_2_1)
  E_right = impurity_entropy(branch_2_2)
  E_split = E_left*(len(branch_2_1))/(len(branch_2_1)+len(branch_2_2)) + E_right*(len(branch_2_2))/(len(branch_2_1)+len(branch_2_2))
  return E_split


def info_gain(y,s_condicion): 

  # s_condicion = data['Gender'] == 'Male'
  
  E_d = impurity_entropy(y) #E parent -> usa todos los ejemplos del nodo
 
 # y = nodo padre antes de la divisón
 # s = subconjutnos resultantes leugo de dividir. 2 conjuntos, uno que s´cumple, otro ue no cumple. calcular impureza o entropia
 # suma ponderada(cant de ejemplos cada sub)  de la simpurezas de los subocnjuntos 
  
  s_1_subset= y[s_condicion]
  s_2_subset= y[-s_condicion]

  s_1_count = len(s_1_subset) #calcula cant de una divi
  s_2_count = len(s_2_subset) #cant de otra div

  if s_1_count == 0 and s_2_count == 0:
    return 0
 
  if y.dtypes != 'O' :
    return info_gain_classification(y, s_1_subset,s_2_subset)
  else:
    return info_gain_regression(y, s_1_subset,s_2_subset)

def info_gain_classification(y, s_1_subset,s_2_subset):
  s_1_count = len(s_1_subset) #calcula cant de una divi
  s_2_count = len(s_2_subset)  
  ponderado_s1 = s_1_count / (s_1_count + s_2_count)
  ponderado_s2 = s_2_count / (s_1_count + s_2_count)

  gain = impurity_entropy(y) - (ponderado_s1 * impurity_entropy(s_1_subset) + ponderado_s2 * impurity_entropy(s_2_subset)) 

  return gain

def info_gain_regression(y, s_1_subset,s_2_subset): 
  s_1_count = len(s_1_subset) #calcula cant de una divi
  s_2_count = len(s_2_subset) 
  ponderado_s1 = s_1_count / (s_1_count + s_2_count)
  ponderado_s2 = s_2_count / (s_1_count + s_2_count)

  gain = y.var() - (ponderado_s1 * s_1_subset.var() + ponderado_s2 * s_2_subset.var()) 

  return gain


def all_split_values(predictor,target):
  all_info_gains = []
  split_value = []

  if predictor.dtypes != 'O':
    options = predictor.sort_values().unique()
 
    for i in options:
      #revisar si cada valor de las opciones (todas las posibildiades dentro de la columna predictor)
      # es mayor a la columna predictora 
      info_gains = info_gain(target, predictor < i)
      all_info_gains.append(info_gains)
      split_value.append(i)


  else:
    options = predictor.unique()
    for i in options: # mask = las entradas con val dentro, se ponen como true 
      # mask ej data['Gender'] == 'Male'
      info_gains = info_gain(target, predictor == i)
      all_info_gains.append(info_gains)
      split_value.append(i)

  return best_split_val(all_info_gains , split_value)

def best_split_val(all_info_gains , split_value):
  if len(all_info_gains) != 0:
    return (split_value[all_info_gains.index(max(all_info_gains))], max(all_info_gains))
  else:
    return False
# Esas funciones la hacen para un target, hacer un loop para que calule
# info gain por predictor y compare y seleccione el de max ganancia

def best_split_fromall(data , y):
  
  column_names = data.columns.tolist()
  all_info_gains =[]
  split_val = []
  for i in range(len(column_names)):
    all_info_gains.append(all_split_values(data[column_names[i]] , y )[1])
    split_val.append(all_split_values(data[column_names[i]],y )[0])
  return (column_names[all_info_gains.index(max(all_info_gains))], split_val[all_info_gains.index(max(all_info_gains))], max(all_info_gains))

def split(data, y):
  best_split_= best_split_fromall(data, y)
  if best_split_ == False:
    return (False)
  else:
    if isinstance(best_split_[1], str):
      data_1 = data[data[best_split_[0]].isin([best_split_[1]])]
      data_2 = data[(data[best_split_[0]].isin([best_split_[1]])) == False]

    else:
      data_1 = data[data[best_split_[0]] < best_split_[1]]
      data_2 = data[data[best_split_[0]] >= best_split_[1]]
  return (data_1,data_2)

def make_prediction(y):
  if isinstance(y,str):
    pred = y.value_counts().idxmax()
  else:
    pred = y.mean()
    
    return pred #moda


def train_tree(data, y, counter):
    if isinstance(data, pd.DataFrame) and len(data) > 0:  # Check if data is a valid DataFrame
        #data = data.drop(columns=[y.name])
        best_split_var, best_split_value, _ = best_split_fromall(data, y)

        if best_split_var is None:  # No valid split found
            pred = make_prediction(y)
            return pred

        # Categorical split
        if isinstance(best_split_value, str):  
            mask = data[best_split_var] == best_split_value  # T / F
            #print(mask)
        else:  # Numerical split
            mask = data[best_split_var] < best_split_value

        data_left = data[mask] #Solo los T
        data_right = data[~mask] #Solo los F
        y_left = y[mask]
        y_right = y[~mask]

        if len(data_left) == 0 or len(data_right) == 0:  # Si una rama está vacía
            pred = make_prediction(y)
            return pred

        counter += 1
        split_type = "in" if isinstance(best_split_value, str) else "<="
        step = "{} {} {}".format(best_split_var, split_type, best_split_value)
        subtree = {step: []}

        yes = train_tree(data_left, y_left, counter)
        no = train_tree(data_right, y_right, counter)

        subtree[step].append(yes)
        subtree[step].append(no)
        return subtree
    else:
        return None


def predict_example(tree, example):
    for condition, sub_tree in tree.items():
        variable, comparison, value = condition.split()
      # Condición categórica
        if comparison == "<=": #Lado yes
            if example[variable] <= float(value):
                if isinstance(sub_tree[0], dict):  # si tiene más nodos
                    return predict_example(sub_tree[0], example) #recursovo bajando de nivel
                else:  # leaf node
                    return sub_tree[0] #regresa clase predicted
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
    predictions = [] 
    for index, row in test_data.iterrows():
        prediction = predict_example(tree, row)
        predictions.append(prediction)

    return predictions

###############################################################

#Upload Dataset

data = pd.read_csv('c:/Users/amb20/OneDrive/Documentos/Ago-Dic23/Person_Gender_Height_Weight_Index.csv') #Change to your own path
data['obese'] = (data.Index >= 4).astype('int')
data.drop('Index', axis = 1, inplace = True)

# PRUEBA 1

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
decision_tree = train_tree(train_data.drop(columns=['obese']), train_data['obese'],0)
predictions = predict_examples(decision_tree, test_data.drop(columns=['obese']))
sklearn.metrics.accuracy_score(test_data['obese'], predictions)
print('Accuracy try 1: ',sklearn.metrics.accuracy_score(test_data['obese'], predictions))


#PRUEBA 2
train_data, test_data = train_test_split(data, test_size=0.2, random_state=2)
decision_tree = train_tree(train_data.drop(columns=['obese']), train_data['obese'],0)
predictions = predict_examples(decision_tree, test_data.drop(columns=['obese']))
print('Accuracy try 2: ',sklearn.metrics.accuracy_score(test_data['obese'], predictions))

#PRUEBA 3
train_data, test_data = train_test_split(data, test_size=0.2, random_state=90)
decision_tree = train_tree(train_data.drop(columns=['obese']), train_data['obese'],0)
predictions = predict_examples(decision_tree, test_data.drop(columns=['obese']))
print('Accuracy try 3: ',sklearn.metrics.accuracy_score(test_data['obese'], predictions))

#PRUEBA 4
train_data, test_data = train_test_split(data, test_size=0.2, random_state=19)
decision_tree = train_tree(train_data.drop(columns=['obese']), train_data['obese'],0)
predictions = predict_examples(decision_tree, test_data.drop(columns=['obese']))
print('Accuracy try 4: ',sklearn.metrics.accuracy_score(test_data['obese'], predictions))