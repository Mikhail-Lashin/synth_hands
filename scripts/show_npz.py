import numpy as np

path = str(input(">>> Input .npz path: "))
data = np.load(path, allow_pickle=True)

print(">>> Num files:", len(data.files))
print(">>> Key examples:", data.files[:3])

# 1st file structure
first_key = data.files[0]
sample = data[first_key].item()

print(f"\n>>> File '{first_key}':")
for key, value in sample.items():
    if hasattr(value, "shape"):
        print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  - {key}: {type(value)}")