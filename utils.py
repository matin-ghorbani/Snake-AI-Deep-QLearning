import matplotlib.pyplot as plt
from IPython import display

plt.ion()


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())

    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    
    last_score = scores[-1]
    last_mean_score = mean_scores[-1]
    plt.text(len(scores) - 1, last_score, str(last_score))
    plt.text(len(mean_scores) - 1, last_mean_score, str(last_mean_score))
    
    plt.show(block=False)
    plt.pause(.1)
