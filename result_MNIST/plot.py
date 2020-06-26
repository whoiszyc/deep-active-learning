import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Random = pd.read_csv("RandomSampling__2020_06_26_15_23.csv")
Margin = pd.read_csv("MarginSampling__2020_06_26_14_14.csv")
Entropy = pd.read_csv("EntropySampling__2020_06_26_16_20.csv")
LeastConfidence = pd.read_csv("LeastConfidence__2020_06_26_15_51.csv")
BALD = pd.read_csv("BALDDropout__2020_06_26_16_46.csv")

Iteration_step = range(len(Random))

plt.plot(Iteration_step, Random, label='Random Sampling', color='black', alpha=1)
plt.plot(Iteration_step, LeastConfidence, label='Uncertainty-LeastConfidence Sampling', color='blue', alpha=1)
plt.plot(Iteration_step, Margin, label='Uncertainty-Margin Sampling', color='red', alpha=1)
# plt.plot(Iteration_step, Margin_1, label='Uncertainty-Margin Sampling', color='darkorange', alpha=1)
plt.plot(Iteration_step, Entropy, label='Uncertainty-Entropy Sampling', color='cyan', alpha=1)
plt.plot(Iteration_step, BALD, label='Bayesian Sampling', color='lime', alpha=1)
plt.xlabel('Iteration Steps')
plt.ylabel('Prediction Accuracy')
plt.legend(title='Query Method:')
plt.grid(color='0.8')
plt.title('Active Learning for MNIST')
plt.show()




