# SSAST: Self-Supervised Audio Spectrogram Transformer
 - [Introduction](#Introduction)
 - [Citing](#Citing)  
 - [Getting Started](#Getting-Started)
 - [SSAST Model](#SSAST-Model) 
 - [Pretraining](#Pretraining)  
 - [Fine-tuning](#Fine-tuning)
 - [Pretrained Models](#Pretrained-Models)
 - [Contact](#Contact)

## Introduction  

<p align="center"><img src="https://github.com/YuanGongND/ssast/blob/main/figure/ssast_ilus.png?raw=true" alt="Illustration of AST." width="300"/></p>

This repository contains the official implementation (in PyTorch) of the **Self-Supervised Audio Spectrogram Transformer (SSAST)** proposed in the AAAI 2022 paper [SSAST: Self-Supervised Audio Spectrogram Transformer](https://arxiv.org/abs/2110.09784) (Yuan Gong, Cheng-I Jeff Lai, Yu-An Chung, James Glass; MIT CSAIL).  

SSAST is the first **patch-based** joint discriminative and generative self-supervised learning framework, and also the first self-supervised learning framework for AST. SSAST significantly boosts AST performance on all downstream tasks we evaluated with an average improvement of 60.9%, leading to similar or even better results than a supervised pretrained AST. SSAST can be used as a drop-in replacement of previous ImageNet (supervised) pretrained AST, and has the advantage of 1) no labeled data is used; 2) flexible patch size and shape, ImagenNet pretraining only supports square patches; and 3) better performance on many tasks, in particular speech tasks.

## Citing  
Please cite our paper if you find this repository useful. 
```  
@article{gong2021ssast,
  title={SSAST: Self-Supervised Audio Spectrogram Transformer},
  author={Gong, Yuan and Lai, Cheng-I Jeff and Chung, Yu-An and Glass, James},
  journal={arXiv preprint arXiv:2110.09784},
  year={2021}
}
```  

  
## Getting Started  

Step 1. Clone or download this repository and set it as the working directory, create a virtual environment and install the dependencies.

```
cd ast/ 
python3 -m venv venvast
source venvast/bin/activate
pip install -r requirements.txt 
```
  

## SSAST Model

The SSAST model script is in ``src/models/ast_models.py``. 

```python
ASTModel(label_dim=527,
         fshape=16, tshape=16 fstride=10, tstride=10,
         input_fdim=128, input_tdim=1024, model_size='base',
         pretrain_stage=True, load_pretrained_mdl_path=None)
```  

**Parameters:**\
`label_dim` : The number of classes, only need to specify in the fine-tuning stage.\
`fshape`: The side length of the patch on the frequency dimension. \
`tshape`: The side length of the patch on the time dimension. \
`fstride`:  The stride of patch spliting on the frequency dimension, for 16\*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6. \
`tstride`:  The stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6. \
`input_fdim`: The number of frequency bins of the input spectrogram.\
`input_tdim`: The number of time frames of the input spectrogram. \
`model_size`: The model size of AST, should be in `[tiny, small, base]` (default: `base`). \
`pretrain_stage`: Set as ``True`` in the self-supervised pretraining stage and ``False`` in the fine-tuning stage. \
`load_pretrained_mdl_path`: The pretrained model used for fine-tuning. Only needed when `pretrain_stage=False` as it is for fine-tuning. 

**Methods:**\
`forward(x, task, cluster=True, mask_patch=400)` \
The entry method of the class that calls fine-tuning and pretraining methods. Parameters:
* `x`: the input spectrogram in shape `[batch_size, time_frame_num, frequency_bin_num]. `Note: the input spectrogram should be normalized with dataset mean and std, see [here](https://github.com/YuanGongND/ast/blob/102f0477099f83e04f6f2b30a498464b78bbaf46/src/dataloader.py#L191).
* `task`: the pretraining or fine-tuning task, should in `[ft_avgtok, ft_cls, pretrain_mpc, pretrain_mpg]`, see below for details.
* `cluster`: set `True` if using cluster patch masking strategy.
* `mask_patch`: the number of patch masked, only needed in the pretraining stage.

`finetuningavgtok(x)`: fine-tune the model by using the average of the outputs of all tokens as the clip represention. Return in shape `[batch_size, label_dim]`.

`finetuningcls(x)`: fine-tune the model by using the output of the `cls` token as clip represention. Return in shape `[batch_size, label_dim]`.

`mpc(x, mask_patch=mask_patch, cluster=cluster)`: pretrain the model with `mask_patch` number of masked patches with the discriminative objective. Return the accuracy and NCE loss of the pretext task.

`mpg(x, mask_patch=mask_patch, cluster=cluster)`: pretrain the model with `mask_patch` number of masked patches with the generative objective. Return the mean square error of the pretext task.

**Example:**
``` python
# pretraining stage
# suppose you have an unlabled dataset with avg length of 1024 frames (i.e., 10.24s)
input_tdim = 1024
# create a 16*16 patch based AST model for pretraining.
# note, we don't use patch split overlap in pretraining, so fstride=fshape and tstride=tshape
ast_mdl = ASTModel(
             fshape=16, tshape=16, fstride=16, tstride=16,
             input_fdim=128, input_tdim=input_tdim, model_size='base',
             pretrain_stage=True)
# # alternatively, create a frame based AST model
# ast_mdl = ASTModel(
#              fshape=128, tshape=2, fstride=128, tstride=2,
#              input_fdim=128, input_tdim=input_tdim, model_size='base',
#              pretrain=True)

# do pretraining, see src/traintest_mask.py for our full pretraining code
# input in shape [batch_size, input_tdim, input_fdim]
test_input = torch.zeros([10, input_tdim, 128])
# mask 100 patches for both discriminative and generative loss
acc, nce_loss = ast_mdl(test_input, task='pretrain_mpc', mask_patch=100)
mse_loss = ast_mdl(test_input, task='pretrain_mpg', mask_patch=100)
loss = nce_loss + 10 * mse_loss
# do back propagate and update the model, etc

# after pretraining, save the pretrained model.
# the code is designed for Dataparallel model
ast_mdl = torch.nn.DataParallel(ast_mdl)
torch.save(ast_mdl.state_dict(), './test_mdl.pth')

# fine-tuning stage
# now you have a labeled dataset you want to finetune AST on
# suppose the avg length is 100 frames (1s) and there are 35 classes
# the fshape and tshape must be same in pretraining and finetuning
# but fstride and tstride can be different in pretraining and finetuning
# using smaller strides improves the performance but also increase the computational overhead
# set pretrain_stage as False since now is in the finetuning stage
# provide the path of the pretrained model you want to load
input_tdim = 100  # fine-tuning data length can be different with pretraining data length
ast_mdl = ASTModel(label_dim=35,
             fshape=16, tshape=16, fstride=10, tstride=10,
             input_fdim=128, input_tdim=input_tdim, model_size='base',
             pretrain_stage=False, load_pretrained_mdl_path='./test_mdl.pth')
# # alternatively, use a frame based AST model
# ast_mdl = ASTModel(label_dim=35,
#              fshape=128, tshape=2, fstride=128, tstride=1,
#              input_fdim=128, input_tdim=input_tdim, model_size='base',
#              pretrain_stage=False, load_pretrained_mdl_path='./test_mdl.pth')

# do finetuning, see src/traintest.py for our finetuning code
test_input = torch.zeros([10, input_tdim, 128])
prediction = ast_mdl(test_input, task='ft_avgtok')
# output should in shape [batch_size, label_dim]
print(prediction.shape)
# calculate the loss, do back propagate, etc
```


## ESC-50 Recipe  
The ESC-50 recipe is in `ast/egs/esc50/run_esc.sh`, the script will automatically download the ESC-50 dataset and resample it to 16kHz, then run standard 5-cross validation and report the result.
The recipe was tested on 4 GTX TITAN GPUs with 12GB memory. 
The result is saved in `ast/egs/esc50/exp/yourexpname/acc_fold.csv` (the accuracy of fold 1-5 and the averaged accuracy), you can also check details in `result.csv` and `best_result.csv` (accuracy, AUC, loss, etc of each epoch / best epoch).
We attached our log file in `ast/egs/esc50/test-esc50-f10-t10-p-b48-lr1e-5`, the model achieves `95.75%` accuracy.

To run the recipe, simply comment out `. /data/sls/scratch/share-201907/slstoolchainrc` in `ast/egs/esc50/run_esc.sh`, adjust the path if needed, and run:
``` 
cd ast/egs/esc50
(slurm user) sbatch run_esc50.sh
(local user) ./run_esc50.sh
```  

## Speechcommands V2 Recipe  
The Speechcommands recipe is in `ast/egs/speechcommands/run_sc.sh`, the script will automatically download the Speechcommands V2 dataset, train an AST model on the training set, validate it on the validation set, and evaluate it on the test set.
The recipe was tested on 4 GTX TITAN GPUs with 12GB memory. 
The result is saved in `ast/egs/speechcommands/exp/yourexpname/eval_result.csv` in format `[val_acc, val_AUC, eval_acc, eval_AUC]`, you can also check details in `result.csv` (accuracy, AUC, loss, etc of each epoch).
We attached our log file in `ast/egs/speechcommends/test-speechcommands-f10-t10-p-b128-lr2.5e-4-0.5-false`, the model achieves `98.12%` accuracy.

To run the recipe, simply comment out `. /data/sls/scratch/share-201907/slstoolchainrc` in `ast/egs/esc50/run_sc.sh`, adjust the path if needed, and run:
``` 
cd ast/egs/speechcommands
(slurm user) sbatch run_sc.sh
(local user) ./run_sc.sh
```  

## Audioset Recipe  
Audioset is a little bit more complex, you will need to prepare your data json files (i.e., `train_data.json` and `eval_data.json`) by your self.
The reason is that the raw wavefiles of Audioset is not released and you need to download them by yourself. We have put a sample json file in `ast/egs/audioset/data/datafiles`, please generate files in the same format (You can also refer to `ast/egs/esc50/prep_esc50.py` and `ast/egs/speechcommands/prep_sc.py`.). Please keep the label code consistent with `ast/egs/audioset/data/class_labels_indices.csv`.

Once you have the json files, you will need to generate the sampling weight file of your training data (please check our [PSLA paper](https://arxiv.org/abs/2102.01243) to see why it is needed).
```
cd ast/egs/audioset
python gen_weight_file.py ./data/datafiles/train_data.json
```

Then you just need to change the `tr_data` and `te_data` in `/ast/egs/audioset/run.sh` and then 
``` 
cd ast/egs/audioset
(slurm user) sbatch run.sh
(local user) ./run.sh
```  
You should get a model achieves `0.448 mAP` (without weight averaging) and `0.459` (with weight averaging). This is the best **single** model reported in the paper. 
The result of each epoch is saved in `ast/egs/audioset/exp/yourexpname/result.csv` in format `[mAP, mAUC, precision, recall, d_prime, train_loss, valid_loss, cum_mAP, cum_mAUC, lr]`
, where `cum_` results are the checkpoint ensemble results (i.e., averaging the prediction of checkpoint models of each epoch, please check our [PSLA paper](https://arxiv.org/abs/2102.01243) for details). The result of weighted averaged model is saved in `wa_result.csv` in format `[mAP, AUC, precision, recall, d-prime]`. We attached our log file in `ast/egs/audioset/test-full-f10-t10-pTrue-b12-lr1e-5/`, the model achieves `0.459` mAP.

In order to reproduce ensembe results of `0.475 mAP` and `0.485 mAP`, please train 3 models use the same setting (i.e., repeat above three times) and train 6 models with different `tstride` and `fstride`, and average the output of the models. Please refer to `ast/egs/audioset/ensemble.py`. We attached our ensemble log in `/ast/egs/audioset/exp/ensemble-s.log` and `ensemble-m.log`. You can use our pretrained models (see below) to test ensemble result.

We use `16kHz` for our experiments. Note that you might get a slightly different result with us due to the YouTube videos are being removed with the time and your downloaded version might be different with us. We provide our evaluation audio ids in `ast/egs/audioset/data/sanity_check/our_as_eval_id.csv`. And please note that in order to compre with the PSLA paper, for the **balanced training set** experiments (with results of `0.347 mAP` and `0.378 mAP`), we use the enhanced label set (i.e., a label set that is modified by an algorithm, please see the PSLA paper for detail). So if you train with the original label set for the balanced training set, you will get a slightly worse result. However, we do not use enhanced label set for **full AudioSet experiments**, i.e., for the `0.459 mAP` (single) and `0.485 mAP` (ensemble) results, we use exactly same data and label with the official release, so you should be able to reproduce that. 

## Pretrained Models
We provide full AudioSet pretrained models and Speechcommands-V2-35 pretrained model.
1. [Full AudioSet, 10 tstride, 10 fstride, with Weight Averaging (0.459 mAP)](https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1)
2. [Full AudioSet, 10 tstride, 10 fstride, without Weight Averaging, Model 1 (0.450 mAP)](https://www.dropbox.com/s/1tv0hovue1bxupk/audioset_10_10_0.4495.pth?dl=1)
3. [Full AudioSet, 10 tstride, 10 fstride, without Weight Averaging, Model 2  (0.448 mAP)](https://www.dropbox.com/s/6u5sikl4b9wo4u5/audioset_10_10_0.4483.pth?dl=1)
4. [Full AudioSet, 10 tstride, 10 fstride, without Weight Averaging, Model 3  (0.448 mAP)](https://www.dropbox.com/s/kt6i0v9fvfm1mbq/audioset_10_10_0.4475.pth?dl=1)
5. [Full AudioSet, 12 tstride, 12 fstride, without Weight Averaging, Model (0.447 mAP)](https://www.dropbox.com/s/snfhx3tizr4nuc8/audioset_12_12_0.4467.pth?dl=1)
6. [Full AudioSet, 14 tstride, 14 fstride, without Weight Averaging, Model (0.443 mAP)](https://www.dropbox.com/s/z18s6pemtnxm4k7/audioset_14_14_0.4431.pth?dl=1)
7. [Full AudioSet, 16 tstride, 16 fstride, without Weight Averaging, Model (0.442 mAP)](https://www.dropbox.com/s/mdsa4t1xmcimia6/audioset_16_16_0.4422.pth?dl=1)

8. [Speechcommands V2-35, 10 tstride, 10 fstride, without Weight Averaging, Model (98.12% accuracy on evaluation set)](https://www.dropbox.com/s/q0tbqpwv44pquwy/speechcommands_10_10_0.9812.pth?dl=1)

If you want to finetune AudioSet-pretrained AST model on your task, you can simply set the `audioset_pretrain=True` when you create the AST model, it will automatically download model 1 (`0.459 mAP`). In our ESC-50 recipe, AudioSet pretraining is used.

If you want to reproduce ensemble experiments, you can download these models at one click using `ast/egs/audioset/download_models.sh`. Ensemble model 2-4 achieves `0.475 mAP`, Ensemble model 2-7 achieves and `0.485 mAP`. Once you download the model, you can try `ast/egs/audioset/ensemble.py`, you need to change the `eval_data_path` and `mdl_list` to run it. We attached our ensemble log in `/ast/egs/audioset/exp/ensemble-s.log` and `ensemble-m.log`.

Please  note that we use `16kHz` audios for training and test (for all AudioSet, SpeechCommands, and ESC-50), so if you want to use the pretrained model, please prepare your data in `16kHz`.

(Note: the above links are Dropbox direct links (i.e., can be downloaded by wget) and should work for most users. For users having issue downloading with the above Dropbox links, it is recommended to use a VPN or use the [OneDrive links](https://mitprod-my.sharepoint.com/:f:/g/personal/yuangong_mit_edu/ErLKkiP-GwVMgdsCeGEjsmoBMtGvXMkX3tCj5_I0E7ikNA?e=JE9Om8), however, OneDrive links are not direct link, please manually download the [`audioset_10_10_0.4593.pth`](https://mitprod-my.sharepoint.com/:u:/g/personal/yuangong_mit_edu/EWrY3raql55CqxZNV3cVSkABaoU7pXQxAeJXudE1PTNzQg?e=gwEICj) and place it in `ast/pretrained_models` if you want to set `audioset_pretrain=True` because the wget link in the `ast/src/models/ast_models.py` would fail if you cannot connect to Dropbox.) 

## Use Pretrained Model For Downstream Tasks

You can use the pretrained AST model for your own dataset. There are two ways to doing so.

You can of course only take ``ast/src/models/ast_models.py``, set ``audioset_pretrain=True``, and use it with your training pipeline, the only thing need to take care of is the input normalization, we normalize our input to 0 mean and 0.5 std. To use the pretrained model, you should roughly normalize the input to this range. You can check ``ast/src/get_norm_stats.py`` to see how we compute the stats, or you can try using our AudioSet normalization ``input_spec = (input_spec + 4.26) / (4.57 * 2)``. Using your own training pipeline might be easier if you already have a good one.
Please note that AST needs smaller learning rate (we use 10 times smaller learning rate than our CNN model proposed in the [PSLA paper](https://arxiv.org/abs/2102.01243)) and converges faster, so please search the learning rate and learning rate scheduler for your task. 

If you want to use our training pipeline, you would need to modify below for your new dataset.
1. You need to create a json file, and a label index for your dataset, see ``ast/egs/audioset/data/`` for an example.
2. In ``/your_dataset/run.sh``, you need to specify the data json file path, the SpecAug parameters (``freqm`` and ``timem``, we recommend to mask 48 frequency bins out of 128, and 20% of your time frames), the mixup rate (i.e., how many samples are mixup samples), batch size, initial learning rate, etc. Please see ``ast/egs/[audioset,esc50,speechcommands]/run.sh]`` for samples.
3. In ``ast/src/run.py``, line 60-65, you need to add the normalization stats, the input frame length, and if noise augmentation is needed for your dataset. Also take a look at line 101-127 if you have a seperate validation set. For normalization stats, you need to compute the mean and std of your dataset (check ``ast/src/get_norm_stats.py``) or you can try using our AudioSet normalization ``input_spec = (input_spec + 4.26) / (4.57 * 2)``.
4. In ``ast/src/traintest.`` line 55-82, you need to specify the learning rate scheduler, metrics, warmup setting and the optimizer for your task.

To summarize, to use our training pipeline, you need to creat data files and modify the above three python scripts. You can refer to our ESC-50 and Speechcommands recipes.

Also, please note that we use `16kHz` audios for the pretrained model, so if you want to use the pretrained model, please prepare your data in `16kHz`.


 ## Contact
If you have a question, please bring up an issue (preferred) or send me an email yuangong@mit.edu.
