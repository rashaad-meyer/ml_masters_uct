# Rashaad Meyer Master's Repository

Hello and Welcome to my master's repo. I am currently developing a frequency-based Deconvolution Layer in PyTorch,
so the TensorFlow and MATLAB folder aren't relevant at this time. Follow the instructions below to run the experiments
that are found in my dissertation:

## Setting up the environment

I would recommend you first create a python virtual environment but it's not needed. You might run into issues if you
don't create one though.

[How to set up virtual environment](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/)

Once you've created and activated your virtual environment. You can install all the necessary packages with the
following command:

```
pip install -r requirements.txt
```

If you'd like the Weights and Biases to track these experiments you'll need to set the `WANDB_API_KEY` environment
variable:

Note: You can get a Weights and Biases API key by signing up and finding it in the settings of their website.

On Linux
```
export WANDB_API_KEY=YOUR_API_KEY
```

On Windows (CMD)
```
set WANDB_API_KEY=YOUR_API_KEY
```

Once you've done this you're ready to start running some of the scripts


## Running the experiments

Make sure you're in the root directory of this project then run one of the following commands depending on which
type of experiment you would like to run

### Single Image Super Resolution

```
python train_regression.py --multi experiment_csv/sisr/srcnn_x2.csv
```

### Image Classification

```
python train_classification.py --multi experiment_csv/img_class/strat.txt
```

### Sound Classification

```
python train_1d_classification.py
```
