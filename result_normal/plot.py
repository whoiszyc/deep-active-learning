import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]  # Same as '-.'

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

# SEED control the results
# Under the same SEED, the same methods will have the same results
# Under the same SEED, Margin and LeastConfidence will have the same results


BALD_1 = pd.read_csv("BALDDropout__2020_06_29_00_36.csv")
BALD_2 = pd.read_csv("BALDDropout__2020_06_29_05_49.csv")
BALD_3 = pd.read_csv("BALDDropout__2020_06_29_09_43.csv")

Entropy_1 = pd.read_csv("EntropySampling__2020_06_29_01_29.csv")
Entropy_2 = pd.read_csv("EntropySampling__2020_06_29_05_03.csv")
Entropy_3 = pd.read_csv("EntropySampling__2020_06_29_08_57.csv")

LeastConfidence_1 = pd.read_csv("LeastConfidence__2020_06_28_21_51.csv")
LeastConfidence_2 = pd.read_csv("LeastConfidence__2020_06_29_03_28.csv")
LeastConfidence_3 = pd.read_csv("LeastConfidence__2020_06_29_07_22.csv")

Margin_1 = pd.read_csv("MarginSampling__2020_06_28_23_36.csv")
Margin_2 = pd.read_csv("MarginSampling__2020_06_29_04_16.csv")
Margin_3 = pd.read_csv("MarginSampling__2020_06_29_08_09.csv")

Random_1 = pd.read_csv("RandomSampling__2020_06_28_22_40.csv")
Random_2 = pd.read_csv("RandomSampling__2020_06_29_02_43.csv")
Random_3 = pd.read_csv("RandomSampling__2020_06_29_06_41.csv")


Iteration_step = range(len(Random_1))

plt.rcParams.update({'font.family': 'Arial'})
plt.figure(figsize=(9, 55))
plt.plot(Iteration_step, Random_1, label='Random Sampling', color='black', alpha=1)
plt.plot(Iteration_step, Random_2, color='black', alpha=1, linestyle=(0, (5, 1)))
# plt.plot(Iteration_step, Random_3, color='black', alpha=0.8)
plt.plot(Iteration_step, LeastConfidence_1, label='Uncertainty-LeastConfidence Sampling', color='blue', alpha=1)
plt.plot(Iteration_step, LeastConfidence_2, color='blue', alpha=1, linestyle=(0, (5, 1)))
# plt.plot(Iteration_step, LeastConfidence_3, color='blue', alpha=0.8)
plt.plot(Iteration_step, Margin_1, label='Uncertainty-Margin Sampling', color='red', alpha=1, linestyle=(0, (5, 10)))
plt.plot(Iteration_step, Margin_2, color='red', alpha=1, linestyle=(0, (1, 1)))
# plt.plot(Iteration_step, Margin_3, color='red', alpha=0.8)
plt.plot(Iteration_step, Entropy_1, label='Uncertainty-Entropy Sampling', color='orange', alpha=1)
plt.plot(Iteration_step, Entropy_2, color='orange', alpha=1, linestyle=(0, (5, 1)))
# plt.plot(Iteration_step, Entropy_3, color='orange', alpha=0.8)
plt.plot(Iteration_step, BALD_1, label='Bayesian Sampling', color='lime', alpha=1)
plt.plot(Iteration_step, BALD_2, color='lime', alpha=1, linestyle=(0, (5, 1)))
# plt.plot(Iteration_step, BALD_3, color='lime', alpha=0.8)
plt.xlabel('Iteration Steps', fontsize=16)
plt.ylabel('Prediction Accuracy', fontsize=16)
plt.legend(fontsize=16)
plt.grid(color='0.8')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Active Learning for Power Flow Solvability - IEEE 39-bus System', fontsize=16)
plt.show()




