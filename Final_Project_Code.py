#!/usr/bin/env python
# coding: utf-8

# In[8]:


import json
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import os
import copy

def percentage_correct(data):
    gpt_o = np.array(data['gpt_results'])
    tst_o = np.array(data['test'][0]['output'])
    counter = 0
    if (len(gpt_o) != len(tst_o)) or (len(gpt_o[0]) != len(tst_o[0])):
        print("Percentage Correct: 0.0")
        print("Count and size:")
        print(0)
        print((len(tst_o) * len(tst_o[0])))
        return
    for v in range(len(gpt_o)):
        for w in range(len(gpt_o[0])):
            if gpt_o[v][w] == tst_o[v][w]:
                counter += 1
    print("Percentage Correct:")
    print(counter/(len(gpt_o) * len(gpt_o[0])))
    print("Count and size:")
    print(counter)
    print((len(tst_o) * len(tst_o[0])))

def load_json_files_from_folder(folder):
    json_files = [file for file in os.listdir(folder) if file.endswith('.json')]
    return {file: json.load(open(os.path.join(folder, file))) for file in json_files}

def create_custom_cmap():
    cvals  = list(range(10))
    colors = ["black", "dodgerblue", "red", "lightgreen", "yellow", "grey", "magenta", "orange", "lightblue", "brown"]
    norm = plt.Normalize(min(cvals), max(cvals))
    return matplotlib.colors.LinearSegmentedColormap.from_list("", list(zip(map(norm, cvals), colors)))

def plot_grid(data, grid_title, axs, cmap):
    axs.set_title(grid_title)
    axs.set_xticks([])
    axs.set_yticks([])
    axs.imshow(np.array(data), cmap=cmap, vmin=0, vmax=9)

def show_results(data):
    cmap = create_custom_cmap()

    fig, axs = plt.subplots(1, 3, figsize=(5, len(data['test']) * 3))
    plot_grid(data['test'][0]['input'], 'Test Input', axs[0], cmap)
    plot_grid(data['test'][0]['output'], 'Test Output', axs[1], cmap)

    if data['gpt_results'] is not None:
        plot_grid(data['gpt_results'], 'GPT Output', axs[2], cmap)
    else:
        axs[2].axis('off')

    fig, axs = plt.subplots(len(data['train']), 2, figsize=(5, len(data['train']) * 3))
    for n, ex in enumerate(data['train']):
        plot_grid(ex['input'], f'Training Input {n}', axs[n, 0], cmap)
        plot_grid(ex['output'], f'Training Output {n}', axs[n, 1], cmap)

    plt.tight_layout()
    plt.show()


# In[9]:


task = '93b581b8.json'
dtaSet = load_json_files_from_folder('/Users/sridonthineni/Documents/CS 221/Final Project/ARC-800-tasks/training')[task]
#put in results from Chat GPT 4 below
dtaSet['gpt_results'] = [[1, 1, 0, 0, 5, 5], [1, 1, 0, 0, 5, 5], [0, 0, 0, 0, 0, 0], [0, 0, 3, 1, 0, 0], [0, 0, 2, 5, 0, 0], [5, 5, 0, 0, 2, 2], [5, 5, 0, 0, 2, 2]]
show_results(dtaSet)
percentage_correct(dtaSet)


# In[ ]:




