# De-identification of facial videos while preserving remote physiological utility
This is the official code repository of our BMVC 2023 paper "De-identification of facial videos while preserving remote physiological utility". Some small intro about our method. 

[Paper](add link), [Poster](add link), [Video](add link)

Add main figure

## Dataset Preprocessing

The original videos are firstly preprocessed to crop and resizethe videos intro T=64 long video segments with 128x128 pixel frames. Facial landmarks are computed with Dlib and saved into a .npy file, for each clip a binary mask is computed based on the landmarks than can be used for traditional methods. The strcuture of the data that can be used with our dataloader is:
01-01
├── blocks (containts [64,128,128] ordered segments of videos)
├──├── 001.npy
├──├── XXX.npy
├── mask (containts [64,128,128] ordered masks of videos, useful for traditional methods)
├──├── 001.npy
├──├── XXX.npy
├── bvp.npy ( array with grountruth bvp signal [T])
├── lnd.npy ( array containing landmarks [T,68,2])

## Pre-training
Please make sure your dataset is processed as described above. 

### Training
Please make sure your dataset is processed as described above.

## Citation

```

