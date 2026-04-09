import matplotlib.pyplot as plt
import numpy as np

# Data
scenarios = ['Urban', 'Field', 'Factory', 'Raceway']
algorithms = ['SORT', 'UCMCTrack', 'SiamRPN', 'DaSiamRPN', 'KCF']
success_rates = [
    0.8066993464, 0.669628268, 0.6651785714, 0.8006535948,
    0.8423202614, 0.7083333333, 0.8611111111, 0.7156862745,
    0.8072916667, 0.9166666667, 0.75, 0.8888888889,
    0.9079861111, 0.9138888889, 0.9117647059, 0.9444444444,
    0.8194444444, 0.7442810458, 0.8333333333, 0.9722222222
]
dos_values = [
    0.9833333333, 0.9861111111, 1, 1,
    1, 0.9861111111, 1, 0.9722222222,
    0.9722222222, 0.9722222222, 0.9444444444, 0.9444444444,
    1, 0.9861111111, 1, 1,
    0.9861111111, 0.9714052288, 0.9444444444, 1,
]

# Reshape data for plotting
success_rates = np.array(success_rates).reshape(5, 4)  # Convert to 0-1 scale
dos_values = np.array(dos_values).reshape(5, 4)  # Convert to 0-1 scale

# Create the bar plot
plt.figure(figsize=(14, 3))

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
plt.ylabel('Success Rate', fontsize=18)
plt.xlabel('Environment Scenarios', fontsize=18)

# Add legend for algorithms and extended bars
from matplotlib.lines import Line2D
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.8, 
                                hatch=hatches[i], edgecolor='black', label=alg)
                  for i, alg in enumerate(algorithms)]
legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.3, 
                                   edgecolor='black', label='Disable'))

plt.legend(handles=legend_elements, fontsize=15, loc='upper center', 
           bbox_to_anchor=(0.5, 1.21), ncol=6, frameon=False)
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0.0, 1.05)

# Set x-axis labels
plt.xticks(x + width * 2, scenarios, fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.show()

# Print the averages for reference
scenario_averages = np.mean(success_rates, axis=0)
print("Scenario averages:")
for scenario, avg in zip(scenarios, scenario_averages):
    print(f"{scenario}: {avg:.3f}")