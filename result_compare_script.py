import numpy as np
from matplotlib import pyplot as plt

base_v = []
base_timing = []
our_v = []
our_timing = []
our_llm_timing = []
plt.figure()
for i in range(1, 6):
    base_v.extend(np.load(f"./DriveLikeAHuman/velocity_trace_{i+5}.npy")[:, 0].tolist())
    our_v.extend(np.load(f"velocity_trace_{i}.npy")[:, 0].tolist())
    base_timing.extend(np.load(f"./DriveLikeAHuman/timing_{i+5}.npy").tolist())
    our_timing.extend(np.load(f"./timing_{i}.npy").tolist())
print(np.mean(base_v), np.std(base_v))
print(np.mean(our_v), np.std(our_v))

our_llm_timing = [our_timing[i] for i in range(len(our_timing)) if i % 5 == 0]
print(np.mean(base_timing), np.std(base_timing))
print(np.mean(our_timing), np.std(our_timing))
print(np.mean(our_llm_timing), np.std(our_llm_timing))
