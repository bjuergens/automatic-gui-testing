# Automatic GUI Testing using the World Model Approach

In this repository the [World Model](https://arxiv.org/abs/1803.10122) approach is adapted for the GUI testing
of desktop software, where only mouse clicks are used. Specifically, a custom implemented desktop software,
called [SUT](https://github.com/neuroevolution-ai/GymGuiEnvironments), which follows the OpenAI Gym environment.

The implementation is based on [world-models](https://github.com/ctallec/world-models) by Tallec, Blier,
and Kalainathan.


## Prerequisites

This implementation was used solely on Linux (Ubuntu and Manjaro), therefore it might not work on Windows, or macOS.
Further required is `xvfb`, to construct a virtual display server. It can be installed on Ubuntu using
`sudo apt install xvfb`.


## Installation

Use Python 3, preferably Python 3.7+, then construct a virtual environment or install directly the packages,
using `pip install -r requirements.txt`. This should also directly install the SUT.


## Using the Approach

Applying the approach requires training the V, M, and C models sequentially. For the V, and M models data is required,
while the C model is trained by interacting with the simulated environment constructed by the M model.

### Data Generation

Use the `data/parallel_data_generation.py` script to generate data in parallel. Call the script with `--help` to
get a list of possible CLI arguments. The script uses random testers in parallel, to generate interaction
sequences between the tester and the SUT. For example to create 10 sequences of 1000 interactions in parallel use
`PYTHONPATH=$(pwd) python data/parallel_data_generation.py -s 10 -p 10 -i --amount 1000 -m random-widgets`

This uses the random widget monkey tester, another option is `-m random-clicks`, which is a normal monkey tester
that uses random coordinates to click through the software.


### Construct Data Sets

#### V Model

Use `data/data_processing/copy_images.py` to copy only the images from the generated sequences, to a folder that is
provided as a CLI argument. Then use `data/data_processing/remove_duplicate_images.py` to deduplicate the images
in that folder. Finally use `data/data_processing/create_dataset_splits.py` to create dataset splits, note that
this last script unfortunately doest not have a CLI interface for the probabilities, so you have to modify its source
(which should be self explanatory).


#### M Model

For the M model dataset, the folders containing the generated sequences can be directly used. Simply construct
a main folder for the dataset, which has three subfolders: `train`, `val`, and `test`. In each of these folders,
create one folder that has the number of interactions of the sequences as its name (the exact name does not matter,
but each subset dir has to have another subfolder). Then copy the sequence folders into this folder. For example:

```
dataset_root_dir
    - train
        - 1000
            - seq_1
            - seq_2
            - seq_3
        - 2000
            - seq_4
    - val 
        - 1000
            - val_seq_1   
    - test
        - 2000
            - test_seq_1
```

### Train V Model

Call `train_vae.py -c PATH_TO_CONFIG`, with a path to a valid VAE config file. An example can be found
in `configs/vae/default_vae_config.yaml`. Copy it and rename it using the prefix `local_`, then Git ignores the file.
There a GPU can be selected by providing the index (this is 0 if there is one GPU, and -1 if there is no CPU). 
Further provide the path to the dataset, as well as other desired hyperparameters. Consider decreasing the batch
size if your GPU has not enough memory. Also you can use different VAE models. The possible choices are defined
in `models/model_selection.py`.

#### Logging using Comet

If you add a file in your home directory called `.comet.config`, which has the following content
```
[comet]
api_key=API_KEY
```
where `API_KEY` is a valid [Comet](https://www.comet.ml/) API key, then the metrics are also
logged to your Comet account. This is also used for the M Model, and C Model training


### Train M Model

Use `train_mdn_rnn.py -c PATH_TO_CONFIG` to train the M Model. Also use the default config
`configs/mdn-rnn/default_mdn_rnn_config.yaml` as a starting point. Again, define the path to the data set,
and also the path to the trained V model there, as well as different hyperparameters. This will automatically use
the V Model to pre-process the necessary data. Subsequent trainings with the same V model and dataset will reuse
the pre-processed data.


### Train C Model

Finally, use `train_controller.py -c PATH_TO_CONFIG` to train the C Model. A default config is available at
`configs/controller/default_controller_config.yaml`. There, define the M model which shall be used for the C model
training. If `evaluate_final_on_actual_environment` is set to `True`, the C model is evaluated on
the SUT after the training, and the log output then shows the achieved code coverage.


Depending on the data set the default hyperparameters might not work. Try varying the batch size and sequence
length to lower values.


## Visualization

The V model logs images during training, which can be seein either in Comet, or if not used by calling
`tensorboard --logdir .` in the log folder.


Further, use `evaluation/dream_visualization/dream_visualization.py` to visualize an M model. There, you can click
through the simulated environment, because the V model is used to visualize the predicted states of the M model.


## Issues and Questions

If you found any bugs or have questions, do not hesitate to create an issue on GitHub. It would be greatly
appreciated!


## Authors

Original Authors:

* **Corentin Tallec** - [ctallec](https://github.com/ctallec)
* **Léonard Blier** - [leonardblier](https://github.com/leonardblier)
* **Diviyan Kalainathan** - [diviyan-kalainathan](https://github.com/diviyan-kalainathan)

Author of this Implementation:

* **Patrick Deubel** - [pdeubel](https://github.com/pdeubel)


## License

I based my implementation on [world-models](https://github.com/ctallec/world-models), which is licensed
under MIT. I also chose this license for this implementation, see [LICENSE](LICENSE).

For the VAE I used the [PyTorch VAE](https://github.com/AntixK/PyTorch-VAE) as inspiration/reference,
copyright A.K Subramanian 2020, licensed under the Apache 2.0 license, see also [LICENSE](LICENSE).
