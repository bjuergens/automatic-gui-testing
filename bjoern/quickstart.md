


installation & vorbereitung

```bash
sudo apt-get install xvfb
# ab hier alles in venv
pip install -U -r requirements.txt
# pip install -U torch==1.10.2 numpy==1.21.6 # bei mir hat torch1.11 probleme gemacht mit dem patcher von comet_ml
export PYTHONPATH=$PWD;$PYTHONPATH

# optional prepare for cpu
# add following line to main-method in train_vae.py: 
torch.set_num_threads(8)
```


einfacher lauf zum debuggen

```bash
# ground truth generieren, ~10min
python data/parallel_data_generation.py -s 10 -p 8 -t --amount=600 --monkey-type=random-clicks --no-log --root-dir=_full_run/01_ground_truth

python data/data_processing/copy_images.py -d _full_run/01_ground_truth/random-clicks/2022-04-21_13-40-06

# count lines --> 13930
ls -l _full_run/01_ground_truth/random-clicks/2022-04-21_13-40-06-mixed | wc -l

# dedup
python data/data_processing/remove_duplicate_images.py -d _full_run/01_ground_truth/random-clicks/2022-04-21_13-40-06-mixed

# count lines --> 238
ls -l _full_run/01_ground_truth/random-clicks/2022-04-21_13-40-06-mixed-deduplicated-images | wc -l

# make splits
python data/data_processing/create_dataset_splits.py -d _full_run/01_ground_truth/random-clicks/2022-04-21_13-40-06-mixed-deduplicated-images 
ls ls _full_run/01_ground_truth/random-clicks/2022-04-21_13-40-06-mixed-deduplicated-images-splits


# tensorboard starten um dem training zuzuschauen (optional)
python -m tensorboard.main --logdir logs --port 8080

# training starten
# ~1h
# python train_vae.py -c configs/myconf.yaml --disable-comet
python train_vae.py -c _full_run/2_vae_config.yaml --disable-comet

# ~2h
python train_mdn_rnn.py -c configs/my_mdn_rnn_conf.yaml --disable-comet


# todo: 
python train_mdn_rnn.py --help
python train_controller.py --help

```


paraleller lauf mit power

```bash
python data/parallel_data_generation.py --number-of-sequences 10 --number-of-processes 8 --amount 20
python data/data_processing/copy_images.py -d datasets/gui_env/random-widgets/2022-03-11_15-35-07 --copy-save-dir datasets/gui_env/random-widgets/allobs
python data/data_processing/create_dataset_splits.py -d datasets/gui_env/random-widgets/allobs
# ggf pfade anpassen in myconf.yaml
python train_vae.py -c configs/myconf.yaml --disable-comet
```


# troubleshooting

## allgemein: hilfe ausgeben

```bash
# diese script laufen auf jeden fall
python data/data_generation.py --help
python data/parallel_data_generation.py --help
python train_vae.py --help
python train_mdn_rnn.py --help
python train_controller.py --help

# diese script existieren
python data/data_processing/calculate_mean_and_std_of_dataset.py --help
python data/data_processing/copy_images.py --help
python data/data_processing/remove_duplicate_images.py --help
python data/data_processing/create_dataset_splits.py --help
python evaluation/visualize_mdn_rnn.py --help
python evaluation/controller/_evaluation_run.py --help
python evaluation/controller/evaluate_controller.py --help
python evaluation/data/visualize_data_sequence.py --help
python train_mdn_rnn_multiple_dataloaders.py --help
```

## ModuleNotFoundError

Fehlermeldung:

    ModuleNotFoundError: No module named 'utils'

Lösung: 

    export PYTHONPATH=$PWD;$PYTHONPATH

