import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def mnist_csv_plot_number(df, index=0, count=None, figszie=(3,3)):

    for i in range(count): 
        image_data = df.iloc[index,1:].values
        label = df.iloc[index,0]

        # Reshape picture to original 28x28 
        image_data = image_data.reshape(28, 28)

        plt.figure(figsize=figszie)
        plt.imshow(image_data, cmap='gray')
        plt.title(f'Row: "{index}", Label: "{label}"')
        
        index +=1

    plt.show()


import numpy as np 
import matplotlib.pyplot as plt

def mnist_csv_plot_number_canvas(df, index=0, figsize=(11,11)): 
    # recommended figsize atleast (11,11) for visbility
    
    images = df.iloc[index,1:].values
    label = df.iloc[index,0]

    # Reshape picture to original 28x28 
    img = images.reshape(28, 28)

    fig = plt.figure(figsize = figsize) 
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            val = round(img[x][y],2) if img[x][y] !=0 else 0
            ax.annotate(str(val), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y]<thresh else 'black')
    
    plt.title(f'Row: "{index}", Label: "{label}"')
    plt.show()