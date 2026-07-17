#! This is the code that works

import json
import time
import cohere
import numpy as np
import pandas as pd

START_ITEM = 0

all_ovon_categories_path = "./data/datasets/goat_openvocab/hm3d/v0.1.2_fixed/val_seen/goat_object_goals.json"
with open(all_ovon_categories_path, "r") as f:
    all_ovon_categories = json.load(f)
all_ovon_categories = sorted(list(set(all_ovon_categories.keys())))
print(f"All OVON categories: {all_ovon_categories}")
items = all_ovon_categories

N = len(items)

API_KEY = "YS0zukcpd9Kyg1TuzHa6e11gh7j3aDOedu6EL1OD" 
co = cohere.Client(API_KEY)

def get_score(item1, item2):
    prompt = (
        f"In a typical home, how likely are a '{item1}' and a '{item2}' to be located near each other? "
        f"Respond with only a number between 0 and 1, where 1 means very likely and 0 means extremely unlikely."
    )
    response = co.generate(
        model="command-r-plus",
        prompt=prompt,
        max_tokens=200,
        temperature=0.3,
        stop_sequences=["\n\n"]
    )
    text = response.generations[0].text.strip()
    score = float(text)
    return min(max(score, 0.0), 1.0)  # clip to [0, 1]




if START_ITEM > 0:
    similarity_matrix = np.load(f"co_location_matrix{START_ITEM-1}.npy")
else:
    similarity_matrix = np.zeros((N, N))

error = False
for i in range(N):
    for j in range(i, N):
        item1, item2 = items[i], items[j]
        if i == j:
            similarity_matrix[i, j] = 1.0
        else:
            for k in range(3):
                try:
                    score = get_score(item1, item2)
                except Exception as e:
                    print("Failed to get score", e)
                similarity_matrix[i, j] = similarity_matrix[j, i] = score
                print(f"{item1} - {item2}: {score}")
                break
            else:
                # if it runs three times and didn't break, it is an error
                error = True
                break
            time.sleep(2)
    
    if error:
        break
    print(f"Row {i} successfull. Saving progress...")
    np.save(f"co_location_matrix_{i}.npy", similarity_matrix)


# ====== SAVE OUTPUT ======
df = pd.DataFrame(similarity_matrix, index=items, columns=items)
df.to_csv("co_location_matrix.csv")
np.save("co_location_matrix.npy", similarity_matrix)
print(df.round(2))
