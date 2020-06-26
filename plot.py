import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Random_1 = pd.read_csv("result/RandomSampling__2020_06_26_00_22.csv")
Margin = pd.read_csv("result/MarginSampling__2020_06_25_21_55.csv")
Margin_1 = pd.read_csv("result/MarginSampling__2020_06_26_02_19.csv")
LeastConfidence = pd.read_csv("result/LeastConfidence__2020_06_25_14_38.csv")

Entropy = [0.6362624052512308, 0.8424337735406736, 0.8604751113542236, 0.8589317808861452, 0.8729194342424006,
           0.8595178557474408, 0.9223060092209111, 0.8798448855200438, 0.8789853090568102, 0.8914980073454716,
           0.8932269281862937, 0.89371532390404, 0.92311674611237, 0.9208896616394467, 0.886487067281394,
           0.8885969367820583, 0.9106235836524186, 0.8623603188247245, 0.8765042588106587, 0.9321813706337423,
           0.850130890052356, 0.8908630929124013]

BALD = [0.6362624052512308, 0.8424337735406736, 0.811362037977651, 0.7978823161678519, 0.8873466437446277,
        0.8525924044697976, 0.839034539345159, 0.8433324216613268, 0.855483707118856, 0.8611197937016488,
        0.8856177229038056, 0.8437426740642338, 0.9152633429710089, 0.9162792060639212, 0.8708974759709307,
        0.898667656481988, 0.8670098460576697,0.877412674845667, 0.8865554426818786, 0.8854614362741268,
        0.897329452215363, 0.8757618973196843]

Random = [0.6362624052512308, 0.7858775494256466, 0.8627901070563413, 0.8573591466750019, 0.8464874579979683,
          0.8578475423927483, 0.8816617175900602, 0.8706630460264124, 0.8170469641322184, 0.8472395874032976,
          0.8401773853246854, 0.8451590216456982, 0.8678987262639681, 0.8651832460732984, 0.8918008126904743,
          0.8816128780182856, 0.8391810580604829, 0.8940962725638821, 0.8529733531296397, 0.920987340782996,
          0.882218488708291, 0.8821012737360319]


Iteration_step = range(len(Random))

plt.plot(Iteration_step, Random, label='Random Sampling', color='black', alpha=1)
plt.plot(Iteration_step, Random_1, label='Random Sampling', color='black', alpha=1)
plt.plot(Iteration_step, LeastConfidence, label='Uncertainty-LeastConfidence Sampling', color='blue', alpha=1)
plt.plot(Iteration_step, Margin, label='Uncertainty-Margin Sampling', color='red', alpha=1)
plt.plot(Iteration_step, Margin_1, label='Uncertainty-Margin Sampling', color='darkorange', alpha=1)
plt.plot(Iteration_step, Entropy, label='Uncertainty-Entropy Sampling', color='cyan', alpha=1)
plt.plot(Iteration_step, BALD, label='Bayesian Sampling', color='lime', alpha=1)
plt.xlabel('Iteration Steps')
plt.ylabel('Prediction Accuracy')
plt.legend(title='Query Method:')
plt.grid(color='0.8')
plt.title('Active Learning for Power Flow Solvability - IEEE 39-bus System')
plt.show()




