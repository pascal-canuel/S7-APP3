import numpy as np
import matplotlib.pyplot as plt


def edit_distance(x, y):
    distance = np.zeros((len(x) + 1, len(y) + 1))

    for i in range(len(x) + 1):
        distance[i, 0] = i

    for j in range(len(y) + 1):
        distance[0, j] = j

    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            distance[i, j] = min(
                distance[i - 1, j - 1] if x[i - 1] == y[j - 1] else distance[i - 1, j - 1] + 1,
                distance[i - 1, j] + 1,
                distance[i, j - 1] + 1
            )

    return distance[len(x), len(y)]


def confusion_matrix(true, pred, labels):
    matrix = np.zeros((len(labels), len(labels))).astype(int)

    if len(true) != len(pred):
        raise ValueError("true and pred must have the same length")

    for i in range(len(true)):
        if len(true[i]) != len(pred[i]):
            raise ValueError("words must have the same length")

        for j in range(len(true[i])):
            matrix[labels.index(true[i][j])][labels.index(pred[i][j])] += 1

    # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    fig, ax = plt.subplots()
    ax.imshow(matrix)

    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="w")

    ax.set_title("Confusion matrix")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Target")
    fig.tight_layout()
    plt.show()
