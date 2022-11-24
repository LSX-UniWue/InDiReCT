# InDiReCT: Language-Guided Zero-Shot Deep Metric Learning for Images

Code for our paper "InDiReCT: Language-Guided Zero-Shot Deep Metric Learning for Images" at WACV 2023.

> Common Deep Metric Learning (DML) datasets specify only one notion of similarity, e.g., two images in the Cars196 dataset are deemed similar if they show the same car model. We argue that depending on the application, users of image retrieval systems have different and changing similarity notions that should be incorporated as easily as possible. Therefore, we present Language-Guided Zero-Shot Deep Metric Learning (LanZ-DML) as a new DML setting in which users control the properties that should be important for image representations without training data by only using natural language. To this end, we propose InDiReCT (**I**mage representatio**n**s using **Di**mensionality **Re**duction on **C**LIP embedded **T**exts), a model for LanZ-DML on images that exclusively uses a few text prompts for training. InDiReCT utilizes CLIP as a fixed feature extractor for images and texts and transfers the variation in text prompt embeddings to the image embedding space. Extensive experiments on five datasets and overall thirteen similarity notions show that, despite not seeing any images during training, InDiReCT performs better than strong baselines and approaches the performance of fully-supervised models. An analysis reveals that InDiReCT learns to focus on regions of the image that correlate with the desired similarity notion, which makes it a fast to train and easy to use method to create custom embedding spaces only using natural language.

## Setup

1. Install all dependencies given in the `requirements.txt`. Additionally, install the CLIP model from https://github.com/openai/CLIP.
2. Download the (test) datasets you are interested in to the `data` folder (the exact paths we used in our experiments can be found in the respective `experiments` notebooks). The datasets are: [Synthetic Cars](https://github.com/konstantinkobs/DML-analysis/tree/master/3D_cars), [Cars196](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), [InShop](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html), [CUB200](http://www.vision.caltech.edu/datasets/cub_200_2011/), [Movie Posters](https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/)

The notebooks in the `experiments` and `analysis` folders contain the experiments and analyses we performed in the paper.
Some lines in the notebooks need to be uncommented during the first run in order to compute the CLIP image embeddings for the images in the data folders.
After they have been created, their embeddings can be reused for all experiments and analyses.
