import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm):

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()

    return fig, ax
