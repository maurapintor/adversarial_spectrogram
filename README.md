# Adversarial spectrograms - Experiments with keyword spotting dataset

Evasion attacks against models for keyword-spotting.

## Getting Started

Get a copy of the repository 

```shell script
git clone https://github.com/Maupin1991/adversarial_spectrogram.git
```

### Prerequisites

Before running the code, we should install the required Python dependencies.

```shell script
pip install -r requirements.txt
```
    

After installing the dependencies, we can go ahead and download and prepare the dataset.
You can download [Google Speech Commands Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) 
from [here](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz). 
After that, we have to split the dataset into training, validation and test set. You can use `prepare_dataset.py`, 
after changing the lines that specify the paths. Remember where you store the dataset because we have to set it in 
the config file. 

```python
src_dir = 'speech_commands_dataset'  # the downloaded and extracted dataset
dst_dir = 'speech_commands_prepared'  # the folder for the splitted dataset
```

Then run:

```shell script
python prepare_dataset.py
```

You should see the dataset split in 3 folders, namely 'train', 'validation' and 'test', in the `dst_dir`.

### Installing

Now let's open the configuration file `config.ini`.

```ini
[TRAINING]
log_interval = 100
scheduler_steps = 20,40
epochs = 30
use_cuda = yes
learning_rate=0.001
weight_decay=0
gradient_penalty=0
lr_decay=0.1
batch_size=64
n_workers=2
[DATA]
data_dir = data/speech_commands_prepared
include_dirs = down,go,left,no,off,on,right,stop,up,yes
n_test=1000
[RESULTS]
model_dir=data/models
plot_dir=data/plots
```

The sections are `TRAINING`, `DATA` and `RESULTS`. 

In the `TRAINING` section:

|NAME|DESCRIPTION|DEFAULT|
|---|---|---|
|`log_interval`|_int_ - number of batches to skip between logging updates during training|100|
|`scheduler_steps`|_ints separated by commas_ (without spaces) - epochs steps where to reduce the learning rate by `lr_decay` (see after)|20,40|
|`epochs`|_int_ - number of epochs to train the network|30|
|`use_cuda`|_bool_ - whether to use GPU, if available|True|
|`learning_rate`|_float_ - starting learning rate to use for training|0.001|
|`weight_decay`|_float_ - weight decay for the optimizer to use during training|0.0|
|`gradient_penalty`|_float_ - penalty for gradients wrt inputs |0.0|
|`lr_decay`|_float_ - multiplier for the learning rate, will decrease the learning rate in the epochs marked as steps (`scheduler_steps`)|0.0001|
|`batch_size`|_int_ - number of samples for each batch|64|
|`n_workers`|_int_ - number of workers to use for loading the dataset|2|

In the `DATA` section:

|NAME|DESCRIPTION|DEFAULT|
|---|---|---|
|`data_dir`|_str_ - path where to find the prepared dataset|~/data/SPEECH|
|`include_dirs`|_strings separated by commas_ (without spaces) - keywords (from the dataset) to include during training|down,go,left,no,off,on,right,stop,up,yes|
|`n_test`|_int_ - number of samples to use for sec eval and creation of adversarial audios|1000|

In the `RESULTS` section:

|NAME|DESCRIPTION|DEFAULT|
|---|---|---|
|`model_dir`|_str_ - path where to store (and later find) the trained models|data/models|
|`plot_dir`|_str_ - path to store the plots|data/plots|

## Examples

### Train a model

After configuring the training parameters, we can already start with training a model. This can be easily done 
with `train_model.py`. Note that you can train multiple models at the same time specifying different parameters in 
the method call (_e.g._ `gradient_penalty`).

```shell script
python train_model.py
```

This will train the CNN defined in `src/net.py` with the parameters stored in the `config.ini` file.

### Evaluate model

If you just want to evaluate the model, you can run: 
```shell script
python evaluate_model.py
```

Note that you have to pass the model name and the loader you want to evaluate the model with. This will be useful 
in the following sections.

### Create adversarial spectrogram and convert back to audio

The adversarial spectrogram is obtained from a spectrogram in the training set, adding a 
worst-case specific perturbation that is called _adversarial_. The adversarial perturbation causes 
the model to fail its classification, resulting in the misclassification of the sample. 
This attack is called **evasion attack**.

There are many techniques for crafting adversarial examples, and here we will use techniques 
called _gradient-based_, that exploit the gradient information for maximizing the target function 
of the attack (sometimes simply the loss of the model). 

Originally, these attacks were tested on images, creating the famous [*adversarial examples*](TODO). 
In this case, we will not limit to the image domain. We will be still using an image and 
computing the perturbation on the image, but later we are also interested to convert back 
the signal (melspectrogram) to the audio domain and listen to the perturbed audio.

In the first attempt we obtained a successful attack, but the perturbation was highly 
identifiable in the audio. The problem stemmed from the added padding zeros of the audio in order 
to have same length for all the samples. Adding the zeros in the original audio resulted 
in adding some _silent_ microsecond to the audio, but after computing the perturbation and 
converting back, we had that what before was silent now sounded like a [_star wars' stormtrooper_](TODO) gun battle.


### Run a security evaluation



## Authors

* **Maura Pintor** - *University of Cagliari* - [maupin1991](https://maupin1991.github.io/)