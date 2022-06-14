# AutoML

LabReport: [Deep Reinforcement Learning for ENAS](https://blog.lanly.vip/article/12)

Deep Reinforcement Learning for Efficient Neural Architecture Search (ENAS) in PyTorch, i.e., AutoML. Code based on the paper [https://arxiv.org/abs/1802.03268](https://arxiv.org/abs/1802.03268) and project [https://github.com/RualPerez/AutoML](https://github.com/RualPerez/AutoML)

How to run:

1. Clone the whole repository
`git clone https://github.com/Lanly109/ENAS.git`

2. Install libraries
`pip3 install -r requirements.txt`

3. Run the main script, for instance:
`python3 main.py --batch 4 --max_layer 6`

Note that you can get a help of how to run the main script by:
`python3 main.py -h`

Once the whole steps have run successfully, the next times you only need to run the last step 3. 

**Output**: The main script saves the trained policy/controller net as policy.pt

Note: If you are using `python3.10`, torch package can't be installed through `pip3`(not know for `conda`). From this [issue](https://github.com/pytorch/pytorch/issues/66424), the author said `pytorch 3.11` will support `python3.10`, and will release soon. However, if you use `pacman` package manager, you can install pytorch through 

```bash
sudo pacman -S python-pytorch       # not support cuda
sudo pacman -S python-pytorch-cuda  # support cuda
``` 

# File structure

```bash
.
├── README.md                       // readme
├── childNet.py                     // childNet source code
├── NLP
│   ├── init.py                     // NLP prepare data program
│   ├── normolize.py                // NLP prepare data program
│   ├── requirements.txt
│   ├── README.md
│   ├── word2vec.json               // NLP training data
│   └── word2vec.py                 // NLP prepare data program
├── img                             // training result visualization
│   ├── accuracy.png                    
│   ├── loss.png                        
│   └── reward.png 
├── log                             // training log
│   ├── draw.py                     // Visualize result program
│   ├── reward.logg
│   ├── reward_nor.logg
│   ├── reward_ori.logg
│   ├── train.loggg
│   ├── train_nor.logg
│   └── train_ori.logg
├── main.py                         // main source code
├── memory.py                       // memory replay source code
├── policy.pt                       // Controller trained parameters
├── policy.py                       // Controller source code
├── requirements.txt
├── sumtree.py                      // memory replay data structure
├── training.py                     // Controller training source code
└── utils.py                        // useful function
```
