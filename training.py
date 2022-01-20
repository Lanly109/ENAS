import torch
from torch.cuda import memory
from childNet import ChildNet
from utils import fill_tensor, indexes_to_actions
from torch.autograd import Variable
import numpy as np
from memory import ReplayMemory
from tqdm import tqdm

def training(policy, batch_size, total_actions, device, verbose = False, num_episodes = 500):
    ''' Optimization/training loop of the policy net. Returns the trained policy. '''
    
    # training settings
    decay = 0.9
    training = True
    val_freq = 10
    
    # childNet & build search space
    cn = ChildNet(policy.layer_limit, total_actions, device)

    # memory capacity
    Memory_size = 50
    # memory replay frequence
    Memory_replay = 20
    memory_pool = ReplayMemory(policy.layer_limit, Memory_size, device)

    # childNet train epochs
    nb_epochs = 100
    
    # train policy network
    training_rewards, val_rewards, losses = [], [], []
    # baseline use for better training
    baseline = torch.zeros(15, dtype=torch.float)

    # record log
    f = open("train.log", "w")
    ff = open("reward.log", "w")
    
    print('start training')
    for i in tqdm(range(num_episodes)):
        if i % 100 == 0: print('Epoch {}'.format(i))

        batch_r, batch_a_probs = [], []
        # forward pass

        # don't process the gradient automaticly. It will be processed manually
        with torch.no_grad():
            # get the result
            if i > Memory_replay and i % Memory_replay == 0:
                m_index, prob, actions, m_reward, _ = memory_pool.sample(batch_size)
            else:
                # action will be an array containing number means the index of total_actions
                # [1,2,3] means total_actions[1], total_actions[2], total_actions[3]
                prob, actions = policy(training)
        # decode the result, show the network
        batch_hid_units, batch_index_eos = indexes_to_actions(actions, batch_size, total_actions)
        
        #compute individually the rewards
        re = []
        # for each data in batch
        for j in range(batch_size):
            # policy gradient update 
            if verbose:
                # print the network of the jth training data
                print(batch_hid_units[j])

            # compute the reward. train the childNet using network componment define by batch_hid_units[j]
            r = cn.compute_reward(batch_hid_units[j], nb_epochs)
            ff.write(f"{r}\n")

            # add in memory
            if i <= Memory_replay or i % Memory_replay:
                memory_pool.push(prob[j], actions[j], r)
            else:
                re.append((r - m_reward[j]).item())

            # the real probabilities
            # because the network created by RNN Controller may contain more than one 'EOS'
            # So just cut off the part after the first EOS
            # it's in the original version code
            # but in my code, it's needless
            a_probs = prob[j, :batch_index_eos[j] + 1]

            # record the reward and probabilities
            batch_r += [r]
            batch_a_probs += [a_probs.view(1, -1)] 

        # update priority of the experience
        if i > Memory_replay and i % Memory_replay == 0:
            memory_pool.batch_update(m_index, np.array([abs(i) if i == i else 0 for i in re]))

        #rearrange the action probabilities for the loss calculation
        # a_probs = []
        # for b in range(batch_size):
        #     a_probs.append(fill_tensor(batch_a_probs[b], policy.n_outputs, ones=True))
        batch_a_probs = torch.stack(batch_a_probs,0)

        #convert to pytorch tensors --> use get_variable from utils if training in GPU
        batch_a_probs = Variable(batch_a_probs, requires_grad=True)
        batch_r = Variable(torch.tensor(batch_r), requires_grad=True)
        
        # classic traininng steps
        loss = policy.loss(batch_a_probs, batch_r, torch.mean(baseline))
        policy.optimizer.zero_grad()  
        loss.backward()
        policy.optimizer.step()

        # actualize baseline
        baseline = torch.cat((baseline[1:]*decay, torch.tensor([torch.mean(batch_r)*(1-decay)], dtype=torch.float)))
        
        # bookkeeping
        training_rewards.append(torch.mean(batch_r).detach().numpy())
        losses.append(loss.item())
        
        # print training
        if (i+1) % val_freq == 0:
            f.write('{:4d} {:6.2f} {:7.4f}\n'.format(i+1, np.mean(training_rewards[-val_freq:]), np.mean(losses[-val_freq:])))
            print('{:4d}. mean training reward: {:6.2f}, mean loss: {:7.4f}'.format(i+1, np.mean(training_rewards[-val_freq:]), np.mean(losses[-val_freq:])))

    print('done training')  
    f.close()
    ff.close()
 
    return policy
