import matplotlib.pyplot as plt
import numpy as np

# Data
appearance = ['same', 'distinct']
algorithms = ['SiamRPN', 'DaSiamRPN', 'KCF']
success_rates = [
    98.1, 86.3,
    92.6, 84.8,
    77.8, 77.3,
]
parentheses_values = [
    100, 94.9,
    100, 93.4,
    100, 97,
]

# Reshape data for plotting
success_rates = np.array(success_rates).reshape(3, 2) / 100  # Convert to 0-1 scale
parentheses_values = np.array(parentheses_values).reshape(3, 2) / 100  # Convert to 0-1 scale

# Create the bar plot
plt.figure(figsize=(5, 3))

# Set up bar positions
x = np.arange(len(appearance)) * 0.5
width = 0.12  # Same width as other plots
colors = ['#2ca02c', '#d62728', '#9467bd']  # First 3 colors for the 3 algorithms
hatches = ['...', '+++', 'xxx']  # First 3 textures for the 3 algorithms

# Create bars for each algorithm
for i, algorithm in enumerate(algorithms):
    # Main bars (success rates)
    plt.bar(x + i * width, success_rates[i], width, label=algorithm, 
            color=colors[i], alpha=0.8, hatch=hatches[i], edgecolor='black', linewidth=0.5)
    
    # Extended bars for parentheses values (shallower color)
    plt.bar(x + i * width, parentheses_values[i], width, 
            color=colors[i], alpha=0.3, edgecolor='black', linewidth=0.5)

# Customize the plot
plt.xlabel('Appearance Similarity', fontsize=18)
plt.ylabel('Success Rate', fontsize=18)
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0.0, 1.1)

# Set x-axis labels
plt.xticks(x + width, appearance, fontsize=16)
plt.yticks(fontsize=16)

# Add legend for algorithms and extended bars
from matplotlib.lines import Line2D
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.8, 
                                hatch=hatches[i], edgecolor='black', label=alg)
                  for i, alg in enumerate(algorithms)]
legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.3, 
                                   edgecolor='black', label='Disable'))

# plt.legend(handles=legend_elements, fontsize=13, loc='lower left')

plt.tight_layout()
plt.show()