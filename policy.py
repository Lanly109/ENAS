import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

class PolicyNet(nn.Module):
    """Policy network, i.e., RNN controller that generates the different childNet architectures."""

    def __init__(self, batch_size, n_hidden, n_active, layer_limit, device):
        super(PolicyNet, self).__init__()
        
        # parameters
        self.layer_limit = layer_limit
        self.gamma = 1.0
        self.n_hidden = 24
        self.n_hidden_type = n_hidden
        self.n_active_type = n_active
        self.n_outputs = n_hidden + n_active
        self.learning_rate = 1e-2
        self.batch_size = batch_size
        self.__device = device
        
        # Neural Network
        self.lstm = nn.LSTMCell(self.n_outputs, self.n_hidden).to(self.__device)
        self.linear = nn.Linear(self.n_hidden, self.n_outputs).to(self.__device)
        
        # training
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
    def one_hot(self, t, num_classes):
        '''One hot encoder of an action/hyperparameter that will be used as input for the next RNN iteration. '''
        out = np.zeros((t.shape[0], num_classes))
        for row, col in enumerate(t):
            out[row, col] = 1
        return out.astype('float32')

    def sample_action(self, output, training):
        '''Stochasticity of the policy, picks a random action based on the probabilities computed by the last softmax layer. '''
        if training:
            random_array = np.random.rand(self.batch_size).reshape(self.batch_size,1)
            return (np.cumsum(output.detach().cpu().numpy(), axis=1) > random_array).argmax(axis=1) # sample action
        else: #not stochastic
            return (output.detach().cpu().numpy()).argmax(axis=1)
                
    def forward(self, training):
        ''' Forward pass. Generates different childNet architectures (nb of architectures = batch_size). '''
        outputs = []
        prob = []
        actions = np.zeros((self.batch_size, self.layer_limit * 2 - 1))
        action = not None #initialize action to don't break the while condition 
        i = 0
        counter_nb_layers = 0
        
        h_t = torch.zeros(self.batch_size, self.n_hidden, dtype=torch.float).to(self.__device)
        c_t = torch.zeros(self.batch_size, self.n_hidden, dtype=torch.float).to(self.__device)
        action = torch.zeros(self.batch_size, self.n_outputs, dtype=torch.float).to(self.__device)
        
        while counter_nb_layers<self.layer_limit: 

            # feed to lstm
            h_t, c_t = self.lstm(action, (h_t, c_t))
                        
            # active value
            output = self.linear(h_t)

            # According to the artical,
            # when number of layer is odd, hidden units will be feed, thus set the value of action function is 0
            # when number of layer is even, action function will be feed, thus set the value of hidden unit is 0
            # but doesn't say the starting number(0 or 1?).
            # From the photo Fig.4 of ENAS artical, the first output should be the active function.
            # So, it should be 0.
            sample_output = output
            if i & 1:
                sample_output[:, counter_nb_layers:] = 0
                sample_output[:, :counter_nb_layers] = F.softmax(sample_output[:, :counter_nb_layers])
            else:
                sample_output[:, :self.n_hidden_type] = 0
                sample_output[:, self.n_hidden_type:] = F.softmax(sample_output[:, self.n_hidden_type:])
                # go to next statu
                counter_nb_layers += 1

            # choose the action function OR hidden unit index according to the output
            action = self.sample_action(sample_output, training)

            # record the output
            outputs += [output]
            # record the probabilities of the chosen action.
            prob.append(output[np.arange(self.batch_size),action])
            actions[:, i] = action
            # encoding the action for the next input to LSTM
            action = torch.tensor(self.one_hot(action, self.n_outputs)).to(self.__device)
            i += 1
            
        prob = torch.stack(prob, 1)
        outputs = torch.stack(outputs, 1).squeeze(2)
        
        return prob, actions

    def loss(self, action_probabilities, returns, baseline):  
        ''' Policy loss '''
        #T is the number of hyperparameters 
        sum_over_T = torch.sum(torch.log(action_probabilities.view(self.batch_size, -1)), axis=1).to(self.__device)
        subs_baseline = torch.add(returns,-baseline).to(self.__device)
        return torch.mean(torch.mul(sum_over_T, subs_baseline)) - torch.sum(torch.mul (torch.tensor(0.01) * action_probabilities, torch.log(action_probabilities.view(self.batch_size, -1))))
