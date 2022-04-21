


installation & vorbereitung

```bash
sudo apt-get install xvfb
# ab hier alles in venv
pip install -U -r requirements.txt
pip install -U torchvision tensorboard
export PYTHONPATH=$PWD;$PYTHONPATH
```


einfacher lauf zum debuggen

```bash
# ground truth generieren, ~10min
python data/data_generation.py -t --amount=100 --monkey-type=random-widgets --no-log --directory=datasets/gui_env/blah

# AssertionError am Ende kann ignoriert werden, wenn die splits-ordner da sind
# ~10s
python data/data_processing/create_dataset_splits.py -d datasets/gui_env/blah/observations
ls datasets/gui_env/blah/observations-splits/*
ls datasets/gui_env/random-widgets/allobs-splits

# tensorboard starten um dem training zuzuschauen (optional)
python -m tensorboard.main --logdir logs --port 8080

# training starten
# ~1h
python train_vae.py -c configs/myconf.yaml --disable-comet

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

