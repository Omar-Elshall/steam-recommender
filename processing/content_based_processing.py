from config import *
import pandas as pd
import numpy as np
import ast

tags = []
for i in range(1, len(games)):
    tags_string = games.loc[i]['tags']
    print(i)
    if games.isnull().loc[i]['tags']:
        continue
    tags_list = ast.literal_eval(tags_string)
    for each in tags_list:
        if each not in tags:
            tags.append(each)

game_tags = pd.DataFrame(index=games['id'], columns=tags)
print(pd.isnull(games.loc[games['id'] == 31990]['tags'].values[0]))

for i in range(1, len(games)):
    tags_string = ''
    id = game_tags.index[i]
    if games.loc[games['id'] == id]['tags'].empty:
        continue
    if pd.isnull(games.loc[games['id'] == id]['tags'].values[0]):
        continue
    tags_string = games.loc[games['id'] == id]['tags'].values[0]
    tags_list = ast.literal_eval(tags_string)
    for each in tags_list:
        game_tags.loc[id][each] = 1
    print(i)
    print(tags_string)

game_tags.fillna(0, inplace=True)
game_tags.to_csv('processing/processedData/game_tags.csv')
print(game_tags.head())

