import matplotlib.pyplot as plt
import numpy as np


loss = []
reward = []
loss_ori = []
reward_ori = []
loss_orii = []
reward_orii = []

acc = []
acc_ori = []
acc_orii = []

with open("./train.logg", "r") as f:
    for line in f.readlines():
        reward.append(float(line.split(' ')[-2].strip()))
        loss.append(float(line.split(' ')[-1].strip()))

with open("./reward.logg", "r") as f:
    for line in f.readlines():
        acc.append(float(line.strip()))

with open("./train_nor.logg", "r") as f:
    for line in f.readlines():
        reward_ori.append(float(line.split(' ')[-4].strip()))
        loss_ori.append(float(line.split(' ')[-1].strip()))

with open("./reward_nor.logg", "r") as f:
    for line in f.readlines():
        acc_ori.append(float(line.strip()))

with open("./train_ori.logg", "r") as f:
    for line in f.readlines():
        reward_orii.append(float(line.split(' ')[-2].strip()))
        loss_orii.append(float(line.split(' ')[-1].strip()))

with open("./reward_ori.logg", "r") as f:
    for line in f.readlines():
        acc_orii.append(float(line.strip()))

def smooth(data):
    cur = 100
    result = []
    tmp = []
    for i, d in enumerate(data):
        if i < cur:
            tmp.append(d)
        else:
            tmp[i % cur] = d
        result.append(np.mean(tmp))
    return result

reward = smooth(reward)
reward_ori = smooth(reward_ori)
reward_orii = smooth(reward_orii)
acc = smooth(acc)
acc_ori = smooth(acc_ori)
acc_orii = smooth(acc_orii)
loss = smooth(loss)
loss_ori = smooth(loss_ori)
loss_orii = smooth(loss_orii)

plt.figure(1)

x1 = range(len(reward))
x2 = range(len(acc))

l1, = plt.plot(x1, reward)
l2, = plt.plot(x1, reward_ori)
l3, = plt.plot(x1, reward_orii)

plt.legend(handles = [l1, l2, l3], labels=["improve_replay", "improve", "origin"])

plt.xlabel('epos') 
plt.ylabel('reward')
plt.title(f'mean reward',size=20)
plt.savefig(f"reward.png", dpi=300)


plt.figure(2)

l1, = plt.plot(x1, loss)
l2, = plt.plot(x1, loss_ori)
l3, = plt.plot(x1, loss_orii)

plt.legend(handles = [l1, l2, l3], labels=["improve_replay", "improve", "origin"])

plt.xlabel('epos') 
plt.ylabel('loss')
plt.title(f'loss',size=20)
plt.savefig(f"loss.png", dpi=300)


plt.figure(3)

l1, = plt.plot(x2, acc)
l2, = plt.plot(x2, acc_ori)
l3, = plt.plot(x2, acc_orii)

plt.legend(handles = [l1, l2, l3], labels=["improve_replay", "improve", "origin"])

plt.xlabel('epos') 
plt.ylabel('accuracy')
plt.title(f'accuracy',size=20)
plt.savefig(f"accuracy.png", dpi=300)
