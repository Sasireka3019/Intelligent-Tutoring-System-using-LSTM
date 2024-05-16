from pyBKT.models import Model
import pandas as pd
import numpy as np

def determine_proficiency(correct_pred, state_pred):
    # Define proficiency thresholds
    easy_threshold = 0.4
    medium_threshold = 0.7
    
    # Compare values against thresholds
    if correct_pred <= easy_threshold and state_pred <= easy_threshold:
        return [1, 1, 1]  # Easy
    elif correct_pred <= medium_threshold and state_pred <= medium_threshold:
        return [0, 1, 1]  # Medium
    else:
        return [0, 0, 1]  # Hard
    
def predict_output(scores):
    model = Model(seed = 42, num_fits = 1)
    input_data = {'Row': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
              'Anon Student Id': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              'skill_name': ['Array', 'Array', 'Array', 'Array', 'Array', 'Linked List', 'Linked List', 'Linked List', 'Linked List', 'Linked List', 'Stack', 'Stack', 'Stack', 'Stack', 'Stack', 'Queue', 'Queue', 'Queue', 'Queue', 'Queue',  'Tree', 'Tree', 'Tree', 'Tree', 'Tree', 'Graph', 'Graph', 'Graph', 'Graph', 'Graph', 'Searching', 'Searching','Searching','Searching','Searching', 'Sorting', 'Sorting', 'Sorting', 'Sorting', 'Sorting', 'Recursion', 'Recursion', 'Recursion', 'Recursion', 'Recursion', 'DP', 'DP', 'DP', 'DP', 'DP'],
              'Correct First Attempt': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              'Problem Name': ['ARRAY001', 'ARRAY002', 'ARRAY003', 'ARRAY004', 'ARRAY005', 'LINKEDLIST001', 'LINKEDLIST002', 'LINKEDLIST003', 'LINKEDLIST004', 'LINKEDLIST005', 'STACK001', 'STACK002', 'STACK003', 'STACK004', 'STACK005', 'QUEUE001', 'QUEUE002', 'QUEUE003', 'QUEUE004', 'QUEUE005', 'TREE001', 'TREE002', 'TREE003', 'TREE004', 'TREE005', 'GRAPH001', 'GRAPH002', 'GRAPH003', 'GRAPH004', 'GRAPH005', 'SEARCHING001', 'SEARCHING002', 'SEARCHING003', 'SEARCHING004', 'SEARCHING005', 'SORTING001', 'SORTING002', 'SORTING003', 'SORTING004', 'SORTING004', 'RECURSION001', 'RECURSION002', 'RECURSION003', 'RECURSION004', 'RECURSION005', 'DP001', 'DP002', 'DP003', 'DP004', 'DP005']
              }
    input_data['Correct First Attempt'] = scores
    my_df = pd.DataFrame(input_data)
    model.fit(data = my_df)
    preds = model.predict(data = my_df)
    filtered_df = preds.groupby('skill_name').first().reset_index()
    result_df = filtered_df.drop(['Row', 'Anon Student Id', 'Correct First Attempt', 'Problem Name'], axis=1)
    skills_data = result_df.to_dict(orient='index')
    
    print(skills_data)

    prediction_result = []
    for skill_data in skills_data:
        skill = skills_data[skill_data]
        proficiency_level = determine_proficiency(skill["correct_predictions"], skill["state_predictions"])
        prediction_result.extend(proficiency_level)
    
    print(prediction_result)
    return np.array([prediction_result])
    
