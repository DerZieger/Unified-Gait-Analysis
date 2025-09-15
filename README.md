# A Unified Pipeline for Explainable Gait Analysis

This repository contains a pipeline to unify clinical gait recordings into a common system.  
We then train a neural network using the unified data to classify the gait into healthy and sick.  
Additionaly, we employ XAI methods to visualize the impact of the inputs on a statistical shape model to show what was pivotal according to the network.

This code accompanies the paper **A Unified Pipeline for Explainable Gait Analysis** acceptted at [ShapeMI 2025](https://shapemi.github.io)


## Setup

First download the repo:
```
git clone https://github.com/DerZieger/Unified-Gait-Analysis.git
```

Use the ```setup.sh``` to setup a conda environment with the required pip packages as well as clone [VPoser](https://github.com/nghorbani/human_body_prior), [SUPR](https://github.com/ahmedosman/SUPR) and [LRP](https://github.com/rodrigobdz/lrp).  
Then download the SUPR model files from [here](https://supr.is.tue.mpg.de/), the VPoser files from [here](https://smpl-x.is.tue.mpg.de/) and put them into the corresponding folder in the data folder.

## Usage

You first need poses for SUPR. You can either use [3D Body Twin](https://github.com/DerZieger/3DBodyTwin) for that or use the [Shape Optimizer](shape_optimizer.py), which is a heavily reduced version of 3D Body Twin only supporting the C3D and Angle constraint, in conjunction with a [Cohort File](example_cohort.json) and a ```constraints.json``` from 3D Body Twin like:
```
python shape_optimizer.py --cohort ./example_cohort.json --model_folder ./data/ --save_folder ./optimized_poses
```

Then either train a [transformer](train_transformer.py) or a [MLP](train_multi_mlp.py). The dataloader uses the SUPR model to extract joint positions and aligns all gait sequences into a common system. The unified data is then used to train the network like:
```
python train_transformer.py --batch_size 64 --interval_size 128 --epochs 25 --lr 0.0001 --pfe_dim 64 --fc_dim 32 --pfe_head 2 --fc_head 4 --save_folder ./trained_transformer --base_folder ./optimized_poses
```
or 
```
python train_multi_mlp.py --batch_size 64 --interval_size 128 --epochs 25 --lr 0.0001 --pfe_layers 16 32 64 32 16 --fc_layers 32 64 128 64 32 --save_folder ./trained_transformer --base_folder ./optimized_poses --lrp
```

Finally, you can use the [viewer](attention_viewer.py) see look at optimized poses of a gait sequence and use the trained models to classify the gait sequence as well as visualize the importance of the inputs on the SUPR model like:

```
python attention_viewer.py --base_folder ./data/SUPR --model supr_neutral.npy --unconstrained
```
During runtime you can move around using W,A,S,D,F,R,E,Q and the mouse. You have a minimal GUI to load the gait sequence and networks and inspect the gait with the importances.


### Disclaimer

The documentation and comments were mostly generated using AI and later checked, but there still may be some errors.
