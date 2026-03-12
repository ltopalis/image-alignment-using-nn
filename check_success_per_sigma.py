import json
import numpy as np
from matplotlib import pyplot as plt

with open("/home/ltopalis/Desktop/image-alignment-using-nn/pretrained_models/test_results_myYaleCroppedA.json", "r") as f:
    data = json.load(f)


rms_all = np.array(data["test_rms_values"], dtype=float)
test_idxs = np.array(data["test_idxs"], dtype=int)

# Ομαδοποίηση ανά sigma (1..10)

sigma_counts = {str(i): 0 for i in range(1, 11)}
sigma_values = {str(i): [] for i in range(1, 11)}

for idx, value in zip(test_idxs, rms_all):
    s_id = str((idx // 10) % 10 + 1)
    sigma_counts[s_id] += 1
    sigma_values[s_id].append(float(value))

count = []
for i in range(1, 11):
    count.append(sum(1 for v in sigma_values[str(i)] if v <= 3) / 1000)

plt.figure()

x = list(range(1, 11))
plt.plot(x, count, marker='o')

plt.xticks(x)
plt.yticks(np.linspace(0, 1, 11))

plt.xlabel("Point Standard Deviation")
plt.ylabel("Frequency of Convergence")

plt.ylim(0, 1)
plt.xlim(0, 10)

plt.grid(True)   # optional but useful for this type of plot

plt.show()
