from policy import PolicyNet
from training import training 
import warnings
warnings.filterwarnings("ignore")
import argparse
import torch 

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # input parameters
    parser = argparse.ArgumentParser(description='Documentation in the following link: https://github.com/Lanly109/ENAS', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--batch', help='Batch size of the policy (int)', nargs='?', const=1, type=int, default=5)
    parser.add_argument('--max_layer', help='Maximum nb layers of the childNet (int)', nargs='?', const=1, type=int, default=4)
    # parser.add_argument('--possible_hidden_units', default=[1,2,4,8,16,32], nargs='*',
    #                    type=int, help='Possible hidden units of the childnet (list of int)')
    parser.add_argument('--possible_act_functions', default=['Sigmoid', 'Tanh', 'ReLU', 'LeakyReLU'], nargs='*', 
                        type=int, help='Possible activation funcs of the childnet (list of str)')
    parser.add_argument('--verbose', help='Verbose while training the controller/policy (bool)', nargs='?', const=1, 
                        type=bool, default=False)
    parser.add_argument('--num_episodes', help='Nb of episodes the policy net is trained (int)', nargs='?', const=1, 
                        type=int, default=5000)
    args = parser.parse_args()

    graph = [i + 1 for i in range(args.max_layer)]
    
    # parameter settings
    graph += ['EOS']
    total_actions = graph + args.possible_act_functions
    
    # setup policy network
    policy = PolicyNet(args.batch, len(graph), len(args.possible_act_functions), args.max_layer, device)
    
    # train
    policy = training(policy, args.batch, total_actions, device, args.verbose, args.num_episodes)
    
    # save model
    torch.save(policy.state_dict(), 'policy.pt')
