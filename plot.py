import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import os

sb.set()

# Ensure the plots directory exists
output_dir = '../GAN/plots'
os.makedirs(output_dir, exist_ok=True)

data = pd.read_csv('loss_logs.csv')

plt.plot('Iteration', 'Discriminator Loss', data=data)
plt.plot('Iteration', 'Generator Loss', data=data)

plt.legend()
plt.title('Training Losses')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_loss.png'))
plt.show()
