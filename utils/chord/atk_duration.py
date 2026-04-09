import matplotlib.pyplot as plt
import numpy as np

# Data
durations = ['1s', '1.5s', '2s']
algorithms = [ 'SORT', 'SiamRPN']
success_rates = [
0.4871794872, 0.5531914894, 0.75,
0.9795918367, 0.9615384615, 0.9444444444
]
parentheses_values = [
0.9230769231, 1, 1,
1, 0.9807692308, 1
]


# Reshape data for plotting
success_rates = np.array(success_rates).reshape(2, 3)
parentheses_values = np.array(parentheses_values).reshape(2, 3)

# Create the line plot
plt.figure(figsize=(4, 5))

# Colors and markers matching the established style
colors = ['#ff7f0e', '#d62728']
markers = ['o', 's']  # Different markers for each algorithm
linestyles = ['-', '--']  # Different line styles

# Create lines for each algorithm
for i, algorithm in enumerate(algorithms):
    plt.plot(durations, success_rates[i], marker=markers[i], label=algorithm,
            color=colors[i], linewidth=2, markersize=8, linestyle=linestyles[i])

# Customize the plot
plt.xlabel('Attack Duration', fontsize=18)
plt.ylabel('Success Rate', fontsize=18)
plt.grid(True, alpha=0.3)
plt.ylim(0.0, 1.0)

# Set font sizes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Add legend
# plt.legend(fontsize=13, loc='lower left')

plt.tight_layout()
plt.show()
