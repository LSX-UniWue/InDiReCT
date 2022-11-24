import torch
from typing import Tuple, List
import clip
import colorsys
from PIL import Image
import numpy as np
import pandas as pd
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from tqdm import tqdm
from dimensionality_reduction import DimRedRecon, NormalizedSoftmax, LinearAutoencoder, Autoencoder
import webcolors
from sklearn.decomposition import PCA
# imagenet features
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


model_to_name = {
    "\model": "InDiReCT",
    "Rand. transform": "Rand. trans.",
    "Linear Autoencoder": "LAE"
}


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model, preprocess = clip.load("ViT-B/32", device=device)


def create_image_embeddings(
    image_paths: List[str],
    image_ids: List[str],
    output_path: str
) -> None:
    image_features = {}

    with torch.no_grad():
        for img_id, img_path in tqdm(zip(image_ids, image_paths), total=len(image_ids)):
            img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            img_features = model.encode_image(img)

            image_features[img_id] = img_features.squeeze().cpu().detach().numpy()
    
    torch.save(image_features, output_path)


def compute_text_features(texts: List[str]):
    text = clip.tokenize(list(set(texts))).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text).cpu().detach()

    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    return text_features


def compute_accuracies(
    image_features: torch.Tensor,
    labels: list[str],
    texts: list[str] = None,
    text_features: torch.Tensor = None,
    num_components: int = 128,
    include_models: list[str] = None,
) -> Tuple[dict, dict]:
    """Computes the accuracy of the model on the given data.

    Args:
        image_features: Base CLIP features for all images. All vectors should be of length one.
        labels: List of labels for all images.
        accuracy_calculator: Accuracy calculator.
        texts: Similarity descriptions. If None, the labels will be used.
        include_models: List of models to include in the results. If None, all models will be included.
    """
    possible_models = ["random", "clip", "ours", "lae", "randtransform", "oracle", "pca", "ae"]
    if include_models is None:
        include_models = possible_models
    else:
        include_models = [model.lower() for model in include_models]
        for m in include_models:
            assert m in possible_models, f"Unknown model: {m}"

    assert (text_features is not None) or (texts is not None), "Either text_features or texts must be provided"

    accuracy_calculator = AccuracyCalculator(device=device)

    possible_labels = list(set(labels))
    label_mapping = {label: i for i, label in enumerate(possible_labels)}
    int_labels = torch.tensor([label_mapping[label] for label in labels])
    
    if text_features is None:
        print("Creating text features")
        text_features = compute_text_features(texts)

    # Perform dimensionality reduction on text to remove unimportant dimensions (to describe the desired information)
    dimred = DimRedRecon(num_components=num_components)

    results = {}

    if "random" in include_models:
        print("Getting random performance")
        # Random embeddings
        random_embeddings = torch.randn((image_features.shape[0], num_components))
        random_embeddings = random_embeddings / random_embeddings.norm(dim=1, keepdim=True)
        random_baseline = accuracy_calculator.get_accuracy(random_embeddings, random_embeddings, int_labels, int_labels, embeddings_come_from_same_source=True)
        results["Random"] = random_baseline
    
    if "clip" in include_models:
        print("Getting CLIP performance")
        # Just CLIP (using all 512 dimensions)
        clip_only = accuracy_calculator.get_accuracy(image_features, image_features, int_labels, int_labels, embeddings_come_from_same_source=True)
        results["CLIP"] = clip_only
    
    if "ours" in include_models:
        print("Getting optimized CLIP performance")
        # Optimized CLIP
        dimred.fit(text_features)
        norm_scaled_image_features = dimred.transform(image_features)
        clip_optim = accuracy_calculator.get_accuracy(norm_scaled_image_features, norm_scaled_image_features, int_labels, int_labels, embeddings_come_from_same_source=True)
        results["\model"] = clip_optim

    if "randtransform" in include_models:
        print("Getting randomly transformed CLIP performance")
        # Random transformation matrix
        dimred.U = torch.normal(0, 0.1, (num_components, text_features.shape[1]), dtype=torch.float)
        norm_scaled_image_features = dimred.transform(image_features)
        randomly_scaled = accuracy_calculator.get_accuracy(norm_scaled_image_features, norm_scaled_image_features, int_labels, int_labels, embeddings_come_from_same_source=True)
        results["Rand. transform"] = randomly_scaled

    if "oracle" in include_models:
        print("Getting oracle baseline performance")
        # "Oracle": Find the best linear transformation that is possible, i.e. the one that puts together images with the same label and pushes different images apart
        dimred = NormalizedSoftmax(num_components=num_components) # for the oracle, we use another way to compute the lower dimensional embeddings
        norm_scaled_image_features = dimred.fit_transform(image_features, int_labels)
        oracle_performance = accuracy_calculator.get_accuracy(norm_scaled_image_features, norm_scaled_image_features, int_labels, int_labels, embeddings_come_from_same_source=True)
        results["Oracle"] = oracle_performance

    if "lae" in include_models:
        print("Getting Linear Autoencoder performance")
        dimred = LinearAutoencoder(num_components=num_components)
        dimred.fit(text_features)
        norm_scaled_image_features = dimred.transform(image_features)
        autoencoder = accuracy_calculator.get_accuracy(norm_scaled_image_features, norm_scaled_image_features, int_labels, int_labels, embeddings_come_from_same_source=True)
        results["Linear Autoencoder"] = autoencoder

    if "ae" in include_models:
        print("Getting Autoencoder performance")
        dimred = Autoencoder(num_components=num_components)
        dimred.fit(text_features)
        norm_scaled_image_features = dimred.transform(image_features)
        autoencoder = accuracy_calculator.get_accuracy(norm_scaled_image_features, norm_scaled_image_features, int_labels, int_labels, embeddings_come_from_same_source=True)
        results["AE"] = autoencoder

    if "pca" in include_models:
        print("Getting PCA performance")
        try:
            dimred = PCA(n_components=num_components)
            dimred.fit(text_features)
            new_image_features = dimred.transform(image_features)
            pca = accuracy_calculator.get_accuracy(new_image_features, new_image_features, int_labels, int_labels, embeddings_come_from_same_source=True)
        except:
            print("PCA failed, but proceed with None for all metrics")
            pca = {k: None for k in accuracy_calculator.get_curr_metrics()}
        results["PCA"] = pca
    
    return pd.DataFrame(results)



def repeat_n_times(n:int, labels, image_features:torch.Tensor, texts:List[str]=None, text_features:torch.Tensor=None, num_components:int=128, include_models:list=None):
    assert (text_features is not None) or (texts is not None), "Either text_features or texts must be provided"

    if text_features is None:
        print("Creating text features")
        text_features = compute_text_features(texts)

    results_dfs = []
    for i in range(n):
        print(f"Run {i+1}")
        results = compute_accuracies(image_features, labels, text_features=text_features, num_components=num_components, include_models=include_models)
        results_dfs.append(results)

    results_mean = sum(results_dfs) / len(results_dfs)
    results_std = np.sqrt(sum([(df - results_mean)**2 for df in results_dfs]) / len(results_dfs))

    return results_mean, results_std



def get_latex_table(means:pd.DataFrame, stds:pd.DataFrame, caption:str="", label:str=""):
    index_mapping = {
        "AMI": "AMI",
        "NMI": "NMI",
        "mean_average_precision": "MAP",
        "mean_average_precision_at_r": "MAP@R",
        "mean_reciprocal_rank": "MRR",
        "precision_at_1": "Prec@1",
        "r_precision": "R-Prec",
    }

    # copy the dataframe to avoid modifying the original
    means = means.copy()
    stds = stds.copy()

    columns = means.columns
    means.columns = [f"{c}_mean" for c in means.columns]
    stds.columns = [f"{c}_std" for c in stds.columns]
    results = pd.concat([means, stds], axis=1)

    final_results = pd.DataFrame()
    for c in columns:
        final_results[c] = results[f"{c}_mean"].map('{:.3f}'.format) + " $\pm$ " + results[f"{c}_std"].map('{:.3f}'.format)
    
    final_results.index = [index_mapping[i] for i in final_results.index]
    
    columns_format = "@{}r" + "c" * len(final_results.columns) + "@{}"

    return final_results.to_latex(bold_rows=True, escape=False, caption=caption, label=label, column_format=columns_format)



# convert color values to color names
def hsv_to_name(row, hue_column="color_hue", sat_column="color_sat", val_column="color_val"):
    # colorsys expects colors in the range [0, 1]
    rgb = colorsys.hsv_to_rgb((row[hue_column] + 0.5) % 1, 0.5*row[sat_column], 0.5*row[val_column])
    rgb = [int(255*x) for x in rgb]

    try:
        color_name = webcolors.rgb_to_name(rgb)
    except ValueError:
        min_colors = {}
        for key, name in webcolors.CSS21_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - rgb[0]) ** 2
            gd = (g_c - rgb[1]) ** 2
            bd = (b_c - rgb[2]) ** 2
            min_colors[(rd + gd + bd)] = name
        color_name = min_colors[min(min_colors.keys())]

    return color_name


def get_wikipedia_cars():
    # Scrapes all car model names from the Wikipedia page
    car_models = pd.read_html("https://en.wikipedia.org/wiki/List_of_automobile_sales_by_model")
    all_models = []
    for models in car_models:
        if "Automobile" in models.columns:
            all_models.extend(models["Automobile"].tolist())

    all_models = list(set(all_models))
    return all_models
