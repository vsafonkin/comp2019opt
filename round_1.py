from itertools import product
import pandas as pd
import numpy as np
from tqdm import tqdm
from numba import njit, prange
import os

HOME = os.environ["HOME"]

import cost_function_24 as utils
from cost_function_24 import build_cost_function


data = pd.read_csv('data/family_data.csv', index_col='family_id')
sample_submission = pd.read_csv("data/sample_submission.csv", index_col='family_id')
submission = pd.read_csv('data/submission_72019.csv', index_col='family_id')

# Build your "cost_function"
cost_function = build_cost_function(data)

# Run it on default submission file
original = submission['assigned_day'].values
original_score = cost_function(original)

print(cost_function(original))

def stochastic_product_search(top_k, fam_size, original, choice_matrix, 
                              disable_tqdm=False, verbose=0,
                              n_iter=500, random_state=2019):
    """
    original (np.array): The original day assignments.
    
    At every iterations, randomly sample fam_size families. Then, given their top_k
    choices, compute the Cartesian product of the families' choices, and compute the
    score for each of those top_k^fam_size products.
    """
    
    best = original.copy()
    best_score = cost_function(best)
    
    np.random.seed(np.random.randint(100000))

    for i in tqdm(range(n_iter)):
        fam_indices = np.random.choice(range(choice_matrix.shape[0]), size=fam_size)
        changes = np.array(list(product(*choice_matrix[fam_indices, :top_k].tolist())))

        # if 93 not in changes.flatten():
        #     continue
        # print(fam_indices)
        # if 1653 not in fam_indices:
        #     continue

        for change in changes:
            new = best.copy()
            new[fam_indices] = change

            new_score = cost_function(new)

            if new_score < best_score:
                print(f"BINGO!!! new score: {new_score:.2f}")
                best_score = new_score
                best = new

                # sample_submission['assigned_day'] = best
                # temp_score = cost_function(best)
                # sample_submission.to_csv(f'temp/submission_{int(temp_score)}.csv')
        
        if new_score < best_score:
            best_score = new_score
            best = new
    
        if verbose and i % verbose == 0:
            print(f"Iteration #{i}: Best score is {best_score:.2f}")
    
    print(f"Final best score is {best_score:.2f}")
    return best


choice_matrix = data.loc[:, 'choice_0': 'choice_9'].values

# Round 1
best = stochastic_product_search(
    choice_matrix=choice_matrix, 
    top_k=4,
    fam_size=3, 
    original=original, 
    n_iter=1000000,
)

sample_submission['assigned_day'] = best
final_score = cost_function(best)
sample_submission.to_csv(f'{HOME}/.cache/pip/temp_submission.csv')