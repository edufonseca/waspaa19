
## Model-agnostic Approaches to Handling Noisy Labels When Training Sound Event Classifiers

This repository contains the code corresponding to the following paper. If you use this code or part of it, please cite:

> Eduardo Fonseca, Frederic Font, and Xavier Serra, "Model-agnostic Approaches to Handling Noisy Labels When Training
Sound Event Classifiers",  in Proc. IEEE WASPAA 2019, NYC, USA, 2019

The framework comprises all the basic stages: feature extraction, training, inference and evaluation. After loading the FSDnoisy18k dataset [1], log-mel energies are computed and a CNN baseline is trained and evaluated. The code allows to test label smoothing regularization (LSR), mixup and a couple of approaches with noise robust loss functions. Please check our paper for more details. The system is implemented in Keras and TensorFlow 1.x.
This code is designed for experimentation - not optimized for speed.

The FSDnoisy18k dataset is described in [1], and it is available through Zenodo from its companion site: <a href="http://www.eduardofonseca.net/FSDnoisy18k/" target="_blank">http://www.eduardofonseca.net/FSDnoisy18k/</a>. 



## Dependencies
This framework is tested on Ubuntu 17.10 using a conda environment. To duplicate the conda environment:

`conda create --name <envname> --file spec-file.txt`

## Directories and files

`config/` includes a `*.yaml` file with the parameters for the experiment  
`logs/` folder where to include output files per experiment  
`main.py` is the main script  
`data.py` contains the data generators and the code for LSR and mixup  
`feat_extract.py` contains feature extraction code  
`architectures.py` contains the architecture for the baseline system and DenSE model
`utils.py` some basic utilities  
`eval.py` evaluation code  
`losses.py` definition of lq loss 
`loss_time.py` definition of time-dependent noise robust loss function 


## Usage

#### (0) Download the dataset:

Download FSDnoisy18k from Zenodo through the <a href="http://www.eduardofonseca.net/FSDnoisy18k/" target="_blank">dataset companion site</a>, unzip it and locate it in a given directory.

#### (1) Edit `config/*.yaml` file:

The goal is to define the parameters of the experiment. The file is structured with self-descriptive sections. Check the paper for full details. The most important parameters are: 

`ctrl.dataset_path`: path where the dataset is located, eg, `/data/FSDnoisy18k/`.   
`ctrl.train_data`: subset of training data. This paper uses `noisy` ,i.e., the noisy train set of FSDnoisy18k.

`learn.mode`:

  - 0: warm-up based mixup
  - 1: standard training procedure, including LSR, standard mixup, and non time-dependent loss function
  - 2: time-dependent loss function: discard large loss instances from each mini-batch of data dynamically at every iteration
  - 3: prune train set: after *n1* epochs, use checkpoint to predict scores on train set, convert to losses, prune clips associated with large loss values, keep a subset to continue learning

LSR parameters (defined in Sections 2.1 and 4.1):  

  - `learn.LSR`: boolean true/false, to enable
  - `learn.eps_LSR_noisy`: smoothing parameter epsilon
  - `learn.LSRmode`: 'GROUPS2' to enable noise-dependent epsilon (else standard LSR is adopted)
  - `learn.delta_eps_LSR`: delta epsilon

mixup parameters (defined in Sections 2.2 and 4.2):

  - `learn.mixup`: boolean true/false, to enable   
  - `learn.mixup_alpha`: strength of interpolation between examples
  - `learn.mixup_mode`: `intra` or `inter`, for applying mixup to examples of the same batch, or different batches
  - `learn.mixup_warmup_epochs`: warm-up training period without mixup (only enabled when `learn.mode`: 0)

Loss functions parameters (defined in Sections 2.3 and 4.3):

  - `loss.type`: defines loss function. To be decided among:
    - `CCE`: categorical_crossentropy aka cross-entropy loss
    - `lq_loss`: L_q loss
    - `lq_lqmax_time_sudden`: time-dependent loss function described in Section 2.3
    - `lq_lqperc_time_sudden`: time-dependent loss function described in Section 2.3

  - `loss.q_loss`: example of a hyper-parameter (q value) of a loss function in first learning stage

  If `learn.mode = 2`, the following can also be defined:
  
   - `learn.stage1_epoch`: number of epochs for stage1
   - `loss.q_loss2`: example of a hyper-parameter (q value) for loss function in second learning stage
   - `loss.perc_loss2`: percentile {99.0, 96.0, 93.0} used as max threshold such that losses greater than that are discarded in second learning stage

  If `learn.mode = 3`, the following can also be defined:

   - `learn.prune_stage1`: number of epochs before pruning
   - `learn.prune_loss_type`: loss function used to compute losses for pruning
   - `learn.prune_discard_percentile`: percentile {99.25, 98.5, 97.75, 97, 96.25} used as max threshold such that clips associated with greater losses are pruned

Please check the paper for more details. The remaining parameters should be intuitive.


#### (2) Execute the code by:
- activating the conda env 
- run, for instance: `CUDA_VISIBLE_DEVICES=0 KERAS_BACKEND=tensorflow python main.py -p config/params.yaml &> logs/output_file.out`

In the first run, log-mel features are extracted and saved. In the following times, the code detects that there is a feature folder. It *only* checks the folder; not the content. If some feature extraction parameters are changed, the program won’t know it.

#### (3) See results:

You can check the `logs/*.out`. Results are shown in a table (search for `ACCURACY - MICRO`).

 
## Contact

You are welcome to contact me privately should you have any question/suggestion at eduardo.fonseca@upf.edu. You can also create an issue.

## References

[1] Eduardo Fonseca, Manoj Plakal, Daniel P. W. Ellis, Frederic Font, Xavier Favory, Xavier Serra, "Learning Sound Event Classifiers from Web Audio with Noisy Labels", In proceedings of ICASSP 2019, Brighton, UK

