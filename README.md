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

```bash
python main.py -id PATH_TO_DATASET_DIR -is IMAGE_HEIGHT_VALUE IMAGE_WIDTH_VALUE
```