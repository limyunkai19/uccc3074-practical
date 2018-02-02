import matplotlib.pyplot as plt
import numpy as np

def show_thumbnails (X, y, classes, samples_per_class = 10):

    num_classes = len(classes)
    plt.figure(figsize=(15, 15))
    for ci, cls in enumerate(classes):
        idxs = np.flatnonzero(y == ci)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + ci + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X[idx])
            plt.axis('off')
            if i == 0:
                plt.title(cls)

    plt.show()