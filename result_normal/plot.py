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

BALD = pd.read_csv("BALDDropout__2020_07_01_17_14.csv").T
BALD = BALD.append(pd.read_csv("BALDDropout__2020_07_02_11_54.csv").T, ignore_index=True)
BALD = BALD.append(pd.read_csv("BALDDropout__2020_07_05_13_59.csv").T, ignore_index=True)
BALD = BALD.append(pd.read_csv("BALDDropout__2020_07_05_13_59.csv").T, ignore_index=True)
BALD = BALD.append(pd.read_csv("BALDDropout__2020_07_05_13_59.csv").T, ignore_index=True)

Entropy = pd.read_csv("EntropySampling__2020_07_01_13_34.csv").T
Entropy = Entropy.append(pd.read_csv("EntropySampling__2020_07_02_08_14.csv").T, ignore_index=True)
Entropy = Entropy.append(pd.read_csv("EntropySampling__2020_07_05_10_25.csv").T, ignore_index=True)
Entropy = Entropy.append(pd.read_csv("EntropySampling__2020_07_06_05_03.csv").T, ignore_index=True)
Entropy = Entropy.append(pd.read_csv("EntropySampling__2020_07_07_08_12.csv").T, ignore_index=True)

LeastConfidence = pd.read_csv("LeastConfidence__2020_07_01_05_54.csv").T
LeastConfidence = LeastConfidence.append(pd.read_csv("LeastConfidence__2020_07_02_00_39.csv").T, ignore_index=True)
LeastConfidence = LeastConfidence.append(pd.read_csv("LeastConfidence__2020_07_05_02_40.csv").T, ignore_index=True)
LeastConfidence = LeastConfidence.append(pd.read_csv("LeastConfidence__2020_07_05_21_19.csv").T, ignore_index=True)
LeastConfidence = LeastConfidence.append(pd.read_csv("LeastConfidence__2020_07_06_20_08.csv").T, ignore_index=True)

Margin = pd.read_csv("MarginSampling__2020_07_01_09_44.csv").T
Margin = Margin.append(pd.read_csv("MarginSampling__2020_07_02_04_30.csv").T, ignore_index=True)
Margin = Margin.append(pd.read_csv("MarginSampling__2020_07_05_06_34.csv").T, ignore_index=True)
Margin = Margin.append(pd.read_csv("MarginSampling__2020_07_06_01_13.csv").T, ignore_index=True)
Margin = Margin.append(pd.read_csv("MarginSampling__2020_07_07_03_02.csv").T, ignore_index=True)

Random = pd.read_csv("RandomSampling__2020_07_01_02_19.csv").T
Random = Random.append(pd.read_csv("RandomSampling__2020_07_01_21_01.csv").T, ignore_index=True)
Random = Random.append(pd.read_csv("RandomSampling__2020_07_04_23_03.csv").T, ignore_index=True)
Random = Random.append(pd.read_csv("RandomSampling__2020_07_05_17_46.csv").T, ignore_index=True)
Random = Random.append(pd.read_csv("RandomSampling__2020_07_06_13_54.csv").T, ignore_index=True)

Iteration_step = range(Random.shape[1])


plt.rcParams.update({'font.family': 'Arial'})
plt.figure(figsize=(9, 7))
# Random
plt.plot(Iteration_step, Random.mean(), label='Random Sampling', color='black', alpha=1)
plt.fill_between(Iteration_step, Random.mean() - Random.std(), Random.mean() + Random.std(), facecolor='black', alpha=0.3)
# LeastConfidence
plt.plot(Iteration_step, LeastConfidence.mean(), label='Uncertainty-Least Confidence Sampling', color='blue', alpha=1)
plt.fill_between(Iteration_step, LeastConfidence.mean() - LeastConfidence.std(), LeastConfidence.mean() + LeastConfidence.std(),
         color='blue', alpha=0.3)
# Margin
plt.plot(Iteration_step, Margin.mean(), label='Uncertainty-Margin Sampling', color='red', alpha=1)
plt.fill_between(Iteration_step, Margin.mean() - Margin.std(), Margin.mean() + Margin.std(), facecolor='red', alpha=0.3)
# Entropy
plt.plot(Iteration_step, Entropy.mean(), label='Uncertainty-Entropy Sampling', color='orange', alpha=1)
plt.fill_between(Iteration_step, Entropy.mean() - Entropy.std(), Entropy.mean() + Entropy.std(), facecolor='orange', alpha=0.3)
plt.xlabel('Iteration Steps', fontsize=20)
plt.ylabel('Training Prediction Accuracy', fontsize=20)
plt.legend(fontsize=20)
plt.grid(color='0.8')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.title('Active Learning for Power Flow Solvability', fontsize=16)
plt.show()
# plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
plt.tight_layout()


