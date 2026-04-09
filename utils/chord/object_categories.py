import matplotlib.pyplot as plt
import numpy as np

# Data
scenarios = ['Vehicle', 'Pedestrian']
algorithms = ['SORT', 'UCMCTrack', 'SiamRPN', 'DaSiamRPN', 'KCF']
success_rates = [
0.7935729847, 0.6792561469,
0.7554466231, 0.8039215686,
0.8888888889, 0.806712963,
0.9328703704, 0.9004493464,
0.8981481481, 0.7461873638,
]
dos_values = [
0.9907407407, 0.9888888889,
0.9907407407, 0.9907407407,
0.9444444444, 0.9814814815,
0.9907407407, 1,
0.9907407407, 0.9624183007,
]


# Reshape data for plotting
success_rates = np.array(success_rates).reshape(5, 2)  # Convert to 0-1 scale
dos_values = np.array(dos_values).reshape(5, 2)  # Convert to 0-1 scale

# Create the bar plot
plt.figure(figsize=(5, 3))

# Set up bar positions
x = np.arange(len(scenarios)) * 0.8
width = 0.12  # Same width as other plots
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Match algorithm order
hatches = ['', '///', '...', '+++', 'xxx']  # Different textures for each algorithm

# Create bars for each algorithm
for i, algorithm in enumerate(algorithms):
    # Main bars (success rates)
    plt.bar(x + i * width, success_rates[i], width, label=algorithm, 
            color=colors[i], alpha=0.8, hatch=hatches[i], edgecolor='black', linewidth=0.5)
    
    # Extended bars for parentheses values (shallower color)
    plt.bar(x + i * width, dos_values[i], width, 
            color=colors[i], alpha=0.3, edgecolor='black', linewidth=0.5)

# Customize the plot
plt.xlabel('Target Scenarios', fontsize=18)
plt.ylabel('Success Rate', fontsize=18)

# Add legend for algorithms and extended bars
from matplotlib.lines import Line2D
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.8, 
                                hatch=hatches[i], edgecolor='black', label=alg)
                  for i, alg in enumerate(algorithms)]
legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.3, 
                                   edgecolor='black', label='Disable'))

# plt.legend(handles=legend_elements, fontsize=13, loc='upper center', 
#            bbox_to_anchor=(0.5, 1.15), ncol=6, frameon=False)
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0.0, 1.05)

# Set x-axis labels
plt.xticks(x + width * 2, scenarios, fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.show()

