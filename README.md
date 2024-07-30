# MediSim

This is the source code for reproducing the inpatient dataset experiments found in the paper "MediSim: Multi-Granular Simulation for Enriching Longitudinal, Multi-Modal Electronic Health Records"

## Generating the Dataset
This code interfaces with the pubilc MIMIC-III ICU stay database. Before using the code, you will need to apply, complete training, and download the tables referenced in `utils/genMediSim.py` and `utils/genNotes.py` from <https://physionet.org>. From there, generate an empty directory `data/` and `data_notes/` before editing the `mimic_dir` variable in the two files and run them. Finally, apply for and download the MIMIC-CXR-JPG database, generate an empty `data_images/`, and configure then run that file. This will generate all of the relevant data files.

## Training a Model
Next, a model can be training by creating an empt `save/` directory and running all of the desired `train_model.py` scripts. The only requirement is to run `train_base*` scripts before the corresponding `train_ss*` scripts.

## Training Baseline Models
Next, any desired baseline models may be trained by changing your working directory to `temporal_baselines/` or `modality_baselines/` and running the corresponding `{baseline_model}.py` script.

## Evaluating the Model(s)
The baseline scripts will evaluate as a part of their script. For the MediSim and ablation models, run the corresponding `test*` scripts.

## Generate Datasets
To generate simulated/extended dataset, run any desired files in the `generate_datasets/` directory and its subdirectories (for baseline models)

## Evaluate Datasets
To evaluate the utility of enriched data, run `augmentation_prediction_temporal.py` and `augmentation_prediction_modality.py`. These may take awhile as they loop through all of the compared models.

## License
MediSim code and model weights are released under the MIT License. See [LICENSE](LICENSE.txt) for additional details.
