


installation & vorbereitung

```bash
sudo apt-get install xvfb
# ab hier alles in venv
pip install -U -r requirements.txt
# pip install -U torch==1.10.2 numpy==1.21.6 # bei mir hat torch1.11 probleme gemacht mit dem patcher von comet_ml
export PYTHONPATH=$PWD;$PYTHONPATH

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


# daten für m training vorbereiten. 
# auf die observation wird unter folgendem pfad zugegriffen: 
_full_run/03_sequences_m_model/[train,val,test]/$seq_length/$ignored/observations

# d.h man legt folgende ordner an:
mkdir -p _full_run/03_sequences_m_model/{train,val,test}/600

# und dann kopiert man da die rohdaten vom anfang rein, also z.B. so:
cp -r _full_run/01_ground_truth/random-clicks/2022-04-21_13-40-06/ _full_run/03_sequences_m_model/train/600
cp -r _full_run/01_ground_truth/random-clicks/2022-04-21_13-40-06/1 _full_run/03_sequences_m_model/test/600/1
cp -r _full_run/01_ground_truth/random-clicks/2022-04-21_13-40-06/2 _full_run/03_sequences_m_model/test/600/2
cp -r _full_run/01_ground_truth/random-clicks/2022-04-21_13-40-06/3 _full_run/03_sequences_m_model/val/600/3

# bei dem obigen experiment mit 600 schritten pro sequenz, sieht ein pfad kann man alle obs-dirs so auflisten
echo _full_run/03_sequences_m_model/{train,val,test}/600/*/observations|xargs -n1 echo


# ~2h
python train_mdn_rnn.py -c _full_run/3_mdn_rnn_config.yaml --disable-comet

# start dream and klick around until you get a reward, which is a important pre-condition for controller training
python evaluation/dream_visualization/dream_visualization.py -d logs/mdn-rnn/multiple_sequences_varying_length_individual_data_loaders_rnn/version_4

# todo: 
python train_controller.py -c _full_run/4_controller_config.yaml --disable-comet

```


paraleller lauf mit power (WIP)

```bash
python data/parallel_data_generation.py --number-of-sequences 10 --number-of-processes 8 --amount 20
python data/data_processing/copy_images.py -d datasets/gui_env/random-widgets/2022-03-11_15-35-07 --copy-save-dir datasets/gui_env/random-widgets/allobs
python data/data_processing/create_dataset_splits.py -d datasets/gui_env/random-widgets/allobs
# ggf pfade anpassen in myconf.yaml
python train_vae.py -c configs/myconf.yaml --disable-comet
```


# troubleshooting

## ModuleNotFoundError

Fehlermeldung:

    ModuleNotFoundError: No module named 'utils'

Lösung: 

    export PYTHONPATH=$PWD;$PYTHONPATH


## GPU Memory

`watch -n 0.3 nvidia-smi`

## allgemein: hilfe ausgeben

```bash
# diese script laufen auf jeden fall
python data/data_generation.py --help
python data/parallel_data_generation.py --help
python train_vae.py --help
python train_mdn_rnn.py --help
python train_controller.py --help

# diese script existieren, muss ich aber noch ausprobieren
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


## verzeichnisstruktur für mdnrnn training

hier ist eine verzeichnisstruktur, die funktioniert. ich vermute auf `html-report` und `observations-splits` könnte man noch verzichten

```
$ tree _full_run/03_sequences_m_model -d
_full_run/03_sequences_m_model
├── test
│ └── 600
│     ├── 6
│     │ ├── html-report
│     │ ├── observations
│     │ └── observations-splits
│     │     ├── test
│     │     ├── train
│     │     └── val
│     └── 7
│         ├── html-report
│         ├── observations
│         └── observations-splits
│             ├── test
│             ├── train
│             └── val
├── train
│ └── 600
│     ├── 0
│     │ ├── html-report
│     │ ├── observations
│     │ └── observations-splits
│     │     ├── test
│     │     ├── train
│     │     └── val
│     ├── 1
│     │ ├── html-report
│     │ ├── observations
│     │ └── observations-splits
│     │     ├── test
│     │     ├── train
│     │     └── val
│     ├── 2
│     │ ├── html-report
│     │ ├── observations
│     │ └── observations-splits
│     │     ├── test
│     │     ├── train
│     │     └── val
│     ├── 3
│     │ ├── html-report
│     │ ├── observations
│     │ └── observations-splits
│     │     ├── test
│     │     ├── train
│     │     └── val
│     ├── 4
│     │ ├── html-report
│     │ ├── observations
│     │ └── observations-splits
│     │     ├── test
│     │     ├── train
│     │     └── val
│     ├── 5
│     │ ├── html-report
│     │ ├── observations
│     │ └── observations-splits
│     │     ├── test
│     │     ├── train
│     │     └── val
│     ├── 6
│     │ ├── html-report
│     │ ├── observations
│     │ └── observations-splits
│     │     ├── test
│     │     ├── train
│     │     └── val
│     └── 7
│         ├── html-report
│         ├── observations
│         └── observations-splits
│             ├── test
│             ├── train
│             └── val
└── val
    └── 600
        ├── 4
        │ ├── html-report
        │ ├── observations
        │ └── observations-splits
        │     ├── test
        │     ├── train
        │     └── val
        └── 5
            ├── html-report
            ├── observations
            └── observations-splits
                ├── test
                ├── train
                └── val


```

## module die funktionieren

```
$ pip freeze                                                                                                                                                                                       2 ↵
absl-py==1.0.0
aiofiles==0.8.0
anyio==3.5.0
argcomplete==2.0.0
attrs==21.4.0
boto3==1.21.44
botocore==1.24.44
Box2D==2.3.2
cachetools==5.0.0
certifi==2021.10.8
charset-normalizer==2.0.12
click==8.1.2
cloudpickle==2.0.0
cma==3.2.2
comet-ml==3.30.0
configobj==5.0.6
coverage==6.3.2
cycler==0.11.0
Deprecated==1.2.13
dill==0.3.4
dnspython==2.2.1
dulwich==0.20.35
eventlet==0.33.0
everett==3.0.0
fiftyone==0.15.1
fiftyone-brain==0.8.1
fiftyone-db==0.3.0
fonttools==4.32.0
future==0.18.2
glob2==0.7
google-auth==2.6.5
google-auth-oauthlib==0.4.6
greenlet==1.1.2
grpcio==1.44.0
gym==0.23.1
gym-gui-environments @ git+https://github.com/neuroevolution-ai/GymGuiEnvironments.git@c15574bba95d696c6a542bd04645914ec868689a
gym-notices==0.0.6
h11==0.12.0
h5py==3.6.0
httpcore==0.14.7
httpx==0.22.0
idna==3.3
imageio==2.17.0
importlib-metadata==4.11.3
Jinja2==3.1.1
jmespath==1.0.0
joblib==1.1.0
jsonschema==4.4.0
kaleido==0.2.1
kiwisolver==1.4.2
Markdown==3.3.6
MarkupSafe==2.1.1
matplotlib==3.5.1
mongoengine==0.20.0
motor==2.5.1
ndjson==0.3.1
networkx==2.8
npzviewer==0.2.0
numpy==1.22.3
nvidia-ml-py3==7.352.0
oauthlib==3.2.0
opencv-python==4.5.5.64
opencv-python-headless==4.5.5.64
packaging==21.3
pandas==1.4.2
patool==1.12
Pillow==9.1.0
plotly==4.14.3
pprintpp==0.4.0
protobuf==3.20.0
psutil==5.9.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
pymongo==3.12.3
pyparsing==3.0.8
PyQt5==5.15.6
PyQt5-Qt5==5.15.2
PyQt5-sip==12.10.1
pyrsistent==0.18.1
PySide6==6.3.0
PySide6-Addons==6.3.0
PySide6-Essentials==6.3.0
python-dateutil==2.8.2
pytz==2022.1
pytz-deprecation-shim==0.1.0.post0
PyWavelets==1.3.0
PyYAML==6.0
requests==2.27.1
requests-oauthlib==1.3.1
requests-toolbelt==0.9.1
retrying==1.3.3
rfc3986==1.5.0
rsa==4.8
s3transfer==0.5.2
scikit-image==0.19.2
scikit-learn==1.0.2
scipy==1.8.0
semantic-version==2.9.0
shiboken6==6.3.0
six==1.16.0
sniffio==1.2.0
sortedcontainers==2.4.0
tabulate==0.8.9
tensorboard==2.8.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorboardX==2.5
threadpoolctl==3.1.0
tifffile==2022.4.8
torch==1.11.0
torchinfo==1.6.5
torchvision==0.12.0
tornado==6.1
tqdm==4.64.0
typing_extensions==4.2.0
tzdata==2022.1
tzlocal==4.2
universal-analytics-python3==1.1.1
urllib3==1.26.9
voxel51-eta==0.6.6
websocket-client==1.3.2
Werkzeug==2.1.1
wrapt==1.14.0
wurlitzer==3.0.2
xmltodict==0.12.0
zipp==3.8.0
```
