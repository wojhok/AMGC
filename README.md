# Automatic Music Genre Classifier

Classification of music genre based on spectrogram of music track.

## Data format

Dataset should be in Folder structure format e.g.

```yaml
dataset/
    class1/
        img1.jpg
        img2.jpg
        ...
    class2/
        img1.jpg
        img2.jpg
        ...
    ...
```

## Running training

Before running training set parameters inside [train_config.yaml](configs/train_config.yml).

To start training run command:

```bash
python main.py -y PATH_TO_CONFIG_YAML
```
