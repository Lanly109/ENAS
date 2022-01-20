import torch
from collections import defaultdict
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

activation_functions = {
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(),
    'Linear': nn.Identity()
}

def create_dataset(p_val=0.1, p_test=0.2):
    import numpy as np
    import sklearn.datasets

    # Generate a dataset and plot it
    np.random.seed(0)
    num_samples = 1000

    X, y = sklearn.datasets.make_moons(num_samples, noise=0.2)
    
    train_end = int(len(X)*(1-p_val-p_test))
    val_end = int(len(X)*(1-p_test))
    
    # define train, validation, and test sets
    X_tr = X[:train_end]
    X_val = X[train_end:val_end]
    X_te = X[val_end:]

    # and labels
    y_tr = y[:train_end]
    y_val = y[train_end:val_end]
    y_te = y[val_end:]

    #plt.scatter(X_tr[:,0], X_tr[:,1], s=40, c=y_tr, cmap=plt.cm.Spectral)
    return X_tr, y_tr, X_val, y_val


def load_NLP_dataset():
    # load NLP dataset
    import json

    with open("NLP/word2vec.json", "r") as f:
        data = json.load(f)

    lables = [test['lable'] for test in data]

    tests = [test['text'] for test in data]

    tests = np.array(tests)

    tests = tests[:1000]
    lables = lables[:1000]

    # training data
    X_tr = np.array(tests[:600])
    y_tr = np.array(lables[:600])
    # testing data
    X_val = np.array(tests[600:])
    y_val = np.array(lables[600:])

    return X_tr, y_tr, X_val, y_val

class Net(nn.Module):

    def __init__(self, num_features, num_classes, layer_limit, total_actions, device): 
        super(Net, self).__init__()
      
        #if hid_units is None or len(hid_units) == 0:
        #    raise Exception('You must specify at least one action!')

        # the dim of input cell
        self.num_features = num_features
        # the dim of output cell
        self.num_classes = num_classes
        # it's on cuda or cpu?
        self.__device = device

        # the dim of hidden cell
        hidden_features = num_features * 2
        
        # max_layers = 7
        # if max_layers < layer_limit:
        #    raise Exception('Maximum layers that ChildNet accepts is '.format(max_layers))

        try:
            # find the graph
            index_eos = total_actions.index("EOS")
            self.graph = total_actions[:index_eos]
            self.node_number = len(self.graph)
        except Exception as e:
            raise Exception("EOS NOT FOUND")

        # input edge, not shown in artical
        input_edge = nn.Linear(num_features, hidden_features).to(self.__device)
        # hidden edge shown in artical
        hidden_edge = [nn.Linear(hidden_features, hidden_features).to(self.__device) for _ in range(layer_limit * (layer_limit - 1) // 2)]
        # output edge, not shown in artical
        output_edge = nn.Linear(hidden_features, num_classes).to(self.__device)

        # combine
        self.net = nn.ModuleList([input_edge, *hidden_edge, output_edge]).to(self.__device)
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)

    def hidden_edge(self, st, ed):
        '''calculate the index of edge from st to ed'''
        # it can be calculated by O(1) using summation formula of isometric series
        inx = ed - st
        cnt = self.node_number - 1
        while st > 1:
            inx += cnt
            cnt -= 1
            st -= 1
        return inx
    
    def forward(self, x, layers):
        ''' build the network according to the layers, etc. chose the active lines'''

        # record the value of the node
        val = defaultdict(dict)
        # record the possible end node
        end_node = set(self.graph)
        
        # record the node having value(check the valid of layers)
        value_node = set()

        # current node
        index_node = 0
        # the input node 0
        val[index_node] = x

        index_node = index_node + 1
        # flow to Node 1 in the graph
        val[index_node] = self.net[0](x.to(self.__device))
        # Node 1 has value now!
        value_node.add(index_node)
    
        for pos, layer in enumerate(layers):
            # layers end, break
            if layer == 'EOS':
                break
            if pos & 1:
                # odd position, it should be Node number
                if isinstance(layer, str):
                    raise Exception("Network WRONG! Expect int but found str")
                if layer not in value_node:
                    raise Exception("Network WRONG! Index Large")

                # data flow from layer to index_node
                val[index_node] = self.net[self.hidden_edge(layer, index_node)](val[layer])
                # now index_node has value
                value_node.add(index_node)
                # the data from layer is translated to index_node, so the layer's data cann't be the result data
                if layer in end_node:
                    end_node.remove(layer)
            else:
                # even position, it should be a activation function 
                if isinstance(layer, int):
                    raise Exception("Network WRONG! Expect str but found int")
                # activate the value of index_node
                val[index_node] = activation_functions[layer](val[index_node])

                # go to the next node
                index_node = index_node + 1

        # just let result be the same variable type as x
        result = val[1]
        for end in end_node:
            # sum up all value of the end node
            result = result + val[end]
        result = result - val[1];

        # calculate the mean
        result = result / len(end_node)

        # calculate the output(classes)
        return self.net[-1](result)
    
def accuracy(ys, ts):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    correct_prediction = torch.eq(ts.long(), torch.max(ys, 1)[1])
    # averaging the one-hot encoded vector
    return torch.mean(correct_prediction.float())
    
def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

class ChildNet():

    def __init__(self, layer_limit, total_actions, device):
        self.criterion = nn.CrossEntropyLoss()

        # prepare the training data and testing data
        X_tr, y_tr, X_val, y_val = load_NLP_dataset()
        self.X_tr = X_tr.astype('float32')
        self.y_tr = y_tr.astype('float32')
        self.X_val = X_val.astype('float32')
        self.y_val = y_val.astype('float32')
        
        # the dim of input
        self.num_features = X_tr.shape[-1]
        self.__device = device
        # the dim of output
        self.num_classes = 2
        # Node number, exclusive Node 0 and Output Node
        self.layer_limit = layer_limit

        # Network need to train
        self.net = Net(self.num_features, self.num_classes, self.layer_limit, total_actions, device)

    def check(self, layers):
        ''' check if layers is valid '''
        index_node = 1
        value_node = set([1])
        for pos, layer in enumerate(layers):
            if layer == 'EOS':
                # the second last should be a activation function
                if not isinstance(layers[pos - 1], str):
                    return False
                else:
                    break
            if pos & 1:
                # odd position, it should be the Node number having value
                if isinstance(layer, str):
                    return False
                if layer not in value_node:
                    return False
                # now index_node has value
                value_node.add(index_node)
            else:
                # even position, it should be a activation function
                if isinstance(layer, int):
                    return False
                index_node += 1
        return True

    def compute_reward(self, layers, num_epochs):
        ''' compute_reward, that is accuracy of the layers given by Controller '''

        if not self.check(layers):
            ''' layers is not valid '''
            raise Exception(f"{layers} it shouldn't happen")

        #print(layers)
        # store loss and accuracy for information
        train_losses = []
        val_accuracies = []
        patience = 10
        
        #print(net)
        max_val_acc = 0
        
        # get training input and expected output as torch Variables and make sure type is correct
        tr_input = Variable(torch.from_numpy(self.X_tr)).to(self.__device)
        tr_targets = Variable(torch.from_numpy(self.y_tr)).to(self.__device)

        # get validation input and expected output as torch Variables and make sure type is correct
        val_input = Variable(torch.from_numpy(self.X_val)).to(self.__device)
        val_targets = Variable(torch.from_numpy(self.y_val)).to(self.__device)

        patient_count = 0

        # training loop
        for _ in range(num_epochs):

            # change the net to train mode
            self.net.train()
            # predict by running forward pass, it will call the forward function of net
            tr_output = self.net(tr_input, layers)
            # compute cross entropy loss
            #tr_loss = F.cross_entropy(tr_output, tr_targets.type(torch.LongTensor)) 
            tr_loss = self.criterion(tr_output.float(), tr_targets.long())
            # zeroize accumulated gradients in parameters
            self.net.optimizer.zero_grad()
            
            # compute gradients given loss
            tr_loss.backward()
            #print(net.l_1.weight.grad)
            # update the parameters given the computed gradients
            self.net.optimizer.step()
            
            train_losses.append(tr_loss.data.cpu().numpy())
        
            #AFTER TRAINING

            # change the net to evaluation mode
            self.net.eval()

            # predict with validation input
            val_output = self.net(val_input, layers)
            val_output = torch.argmax(F.softmax(val_output, dim=-1), dim=-1)
            
            # compute loss and accuracy
            #val_loss = self.criterion(val_output.float(), val_targets.long())
            val_acc = torch.mean(torch.eq(val_output.to(self.__device), val_targets.type(torch.LongTensor).to(self.__device)).type(torch.FloatTensor))
            
            #accuracy(val_output, val_targets)
            val_acc = float(val_acc.numpy())
            val_accuracies.append(val_acc)
            
            
            #early-stopping
            if max_val_acc > val_acc:
                patient_count += 1             
                if patient_count == patience:
                    break
            else:
                max_val_acc = val_acc
                patient_count = 0
            
        #print(val_acc)
        return val_acc#max_val_acc#**3 #-float(val_loss.detach().numpy()) 
