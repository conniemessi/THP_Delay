import matplotlib.pyplot as plt

# Read data from file
filename = 'data/toy_one/3dim_20000seq_64ev/2000_100_20_input3_hidden32_softplus.txt'
epoch = []
log_likelihood = []
accuracy = []
rmse = []
gd_masker = []
gd_thp = []

with open(filename, 'r') as file:
    next(file)  # Skip parameters
    next(file)  # Skip header line if present
    lines = file.readlines()
    for line in lines:
        if "delay" in line:
            continue  # Skip lines containing "delay"
        data = line.strip().split(',')
        # print(data)
        if len(data) == 7:  # Ensure it's a valid data line
            epoch.append(int(data[0]))
            log_likelihood.append(float(data[4]))
            accuracy.append(float(data[2]))
            rmse.append(float(data[3]))
            gd_masker.append(float(data[5]))
            gd_thp.append(float(data[6]))

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(5, 1, 1)
plt.plot(epoch, log_likelihood, marker='o', ms=2, linestyle='-')
plt.title('Log-Likelihood vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Log-Likelihood')
plt.ylim(0)

plt.subplot(5, 1, 2)
plt.plot(epoch, accuracy, marker='o', ms=2, linestyle='-')
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(5, 1, 3)
plt.plot(epoch, rmse, marker='o', ms=2, linestyle='-')
plt.title('RMSE vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('RMSE')

plt.subplot(5, 1, 4)
plt.plot(epoch, gd_masker, marker='o', ms=2, linestyle='-')
plt.title('Gradient_norm Masker vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Gradient_norm Masker')

plt.subplot(5, 1, 5)
plt.plot(epoch, gd_thp, marker='o', ms=2, linestyle='-')
plt.title('Gradient_norm THP vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Gradient_norm THP')

plt.tight_layout()
plt.show()