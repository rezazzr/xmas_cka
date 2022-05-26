# Reliability of CKA as a Similarity Measure in Deep Learning

## Section 4.1:
In order to reproduce the 3 networks seen in this section i.e. Generalized, Memorized, and Randomized, we can
run the script `early_layer_study.py` with `--memorize` flag indicating whether the network is to memorize on the data
generalize on it. Therefore, to produce the Memorized network we run:

```shell 
python early_layer_study.py --data_root "path_to_the_data_directory" --memorize
```
And to produce the Generalized network we run:

```shell 
python early_layer_study.py --data_root "path_to_the_data_directory" --memorize
```
The Random network, needs no training hence all we need to do is to create it and initialize it:
```python
from models.cifar_10_models.vgg import VGG7
from utilities.utils import xavier_uniform_initialize

model = VGG7()
model.apply(xavier_uniform_initialize)
```
To compare their CKA values in order to reproduce Figure 2, we then run the following script:
```shell
python evaluate_cka.py -data_root "path_to_the_data_directory"\
--model_path "path to the first model."\
--second_model_path "path to the first model."\
--output_path "where to save the results. The results will be a .npy file."\
--rbf_sigma "If the RBF sigma is == -1 then the linear CKA is computed, else the kernel CKA is computed and the sigma will indicate the multiplier to the median distance."\
```

## Section 4.2:
In order to reproduce the results shown in Section 4.2, follow these steps:
1. Move to subdirectory `section_4_2`.
2. Run notebooks ``artificial_linear_cka.ipynb``, ``artificial_rbf_cka.ipynb`` and ``cifar10_translations.ipynb`` to generate the data.
3. Run notebook ``figure4.ipynb`` to generate Figure 4 from the data.

## Section 4.3:
In order to reproduce the results seen in Figures 5,6, and 7 we need to do the followings:
1. Generate the _Original Networks_.
2. Generate the networks whose _Maps are optimized w.r.t target CKA map_.

For step 1, depending on the Figure, we either need a ResNet or VGG model, and we may want to change the width
of the model as well. The general script for this type of training is given below:
```shell
python main.py --data_root "path_to_the_data_directory"\
--network_width "Some integer indicating width of the network, default to 1."\
--model_type "Either VGG or ResNet."
```
In order to perform step 2, we need the model from step ⬆ and its CKA map. We can compute and save the CKA
map using the script below:
```shell
python evaluate_cka.py -data_root "path_to_the_data_directory"\
--model_path "path to the first model."\
--network_width "an integer to multiply the width of the base model by. Default is 1."\
--model_type "Either VGG or ResNet"\
--output_path "where to save the results. The results will be a .npy file."\
--rbf_sigma "If the RBF sigma is == -1 then the linear CKA is computed, else the kernel CKA is computed and the sigma will indicate the multiplier to the median distance."\
```
Now that we have both the CKA map and the path to the pretrained models, we can run step 2 via:
```shell
python main.py --with_map\
--data_root "path_to_the_data_directory"\
--model_path "path to the pretrained model from step 1."\
--network_width "The width that was used to produce the model."\
--model_type "Either VGG or ResNet"\
--experiment_name "For Figures 7 this should start with `PretrainedMap` for Figure 5 and 6 the options are listed right after this script."\
--cka_path "path to the CKA map of the pretrained model from step 1."\
--accuracy_upper_bound "In case of optimizing w.r.t another network and taking the said network's accuracy in to account this parameter accounts for the said accuracy upper bound."\
--rbf_sigma "If the RBF sigma is == -1 then the linear CKA is computed, else the kernel CKA is computed and the sigma will indicate the multiplier to the median distance."
```
> 🚨 **--experiment_name** 
> 
> Can accept either of the following values in order to produce the maps seen in Figures 5 and 6.
> * `GoblinCKA` ➡ to produce _Goblin_ map in Figure 6.
> * `CarrotCKA` ➡ to produce _Carrot_ map in Figure 6.
> * `SwordCKA` ➡ to produce _Sword_ map in Figure 6.
> * `BowArrowCKA` ➡ to produce _Bow & Arrow_ map in Figure 6.
> * `XMassTreeCKA` ➡ to produce _Christmas Tree_ map in Figure 6.
> * `AllOnesCKA` ➡ to produce _maximized CKA similarity between all layers_ map in Figure 5.
> * `AllZerosCKA` ➡ to produce _minimized CKA similarity between all layers_ map in Figure 5.
> * `SingleOneCKA` ➡ to produce _maximized CKA similarity between the 1st and last layer_ map in Figure 5.

Once this training is over, we can use the `evaluate_cka.py` again to compute and save the CKA map of the networks for plotting.