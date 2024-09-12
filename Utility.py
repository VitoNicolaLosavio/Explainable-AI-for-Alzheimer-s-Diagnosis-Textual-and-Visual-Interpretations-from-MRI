import os, random, warnings, torch, cv2, textwrap

import numpy as np
import torch
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureTyped,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAxisFlipd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityd,
    Spacingd,
    SpatialPadd
)

from sys import platform
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

from alzheimer_disease.src.helpers.utils import get_device
from alzheimer_disease.src.models.gradcam import GradCAM
from alzheimer_disease.src.modules.preprocessing import get_transformations


def get_gradcam(
        example,
        model,
        saved_path,
        threshold
):
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM), and
    returns a 3D segmentation mask over the `image` brain shape and above a fixed `threshold`.
    Args:
        example (dict): a testing set example.
        model (torch.nn.Module): the model to be loaded.
        saved_path (str): folder path where to find the model dump.
        threshold (int | float): heat threshold above which draw the mask.
        plot_results (bool): whether to plot the visual results.
        alpha (int): transparency channel. Between 0 and 255.
    Returns:
        image (numpy.ndarray): the input 3D image.
        mask (numpy.ndarray): the related computed 3D segmentation mask.
        pred (int): model prediction.
    """
    warnings.filterwarnings('ignore')
    try:
        device = get_device()
        model.load_state_dict(
            torch.load(saved_path + model.name + '_best.pth')
        )
        model.to(device)
        _, eval_transform = get_transformations(model.in_size)
        x = eval_transform([example])
        x_input = [
            torch.unsqueeze(x[0]['image'], 0).to(device),
            torch.unsqueeze(x[0]['data'], 0).to(device)
        ]
        cam = GradCAM(nn_module=model, target_layers='output_layers.relu')
        heatmap = cam(x=x_input)
        model.eval()
        with torch.no_grad():
            pred = model(x_input)
        image = x[0]['image'].squeeze().detach().cpu().numpy()
        label = int(x[0]['label'].squeeze().detach().cpu().numpy())
        pred = int(pred.argmax(dim=1).squeeze().detach().cpu().numpy())
        heatmap = heatmap.squeeze().detach().cpu().numpy()
        mask = _get_heatmap_mask(image, heatmap, threshold)
        return image, mask, pred, label, heatmap
    except OSError as e:
        print(e)
        print('\n' + ''.join(['> ' for i in range(30)]))
        print('\nERROR: model dump for\033[95m ' + model.name + '\033[0m not found.\n')
        print(''.join(['> ' for i in range(30)]) + '\n')


def _get_heatmap_mask(image, heatmap, threshold):
    """
    Computes a 3D segmentation mask over the `image` brain shape,
    according to Grad-CAM `heatmap` and above a fixed `threshold`.
    Args:
        image (numpy.ndarray): the input 3D image.
        heatmap (numpy.ndarray): the Grad-CAM 3D heatmap.
        threshold (int): heat threshold above which draw the mask.
    Returns:
        mask (numpy.ndarray): the computed 3D segmentation mask.
    """
    bg_mask = np.zeros(image.shape, dtype='uint8')
    heatmap_mask = np.zeros(image.shape, dtype='uint8')
    for z in range(image.shape[2]):
        # findind brain contours
        gray = cv2.normalize(image[:, :, z], np.zeros((image.shape[1], image.shape[0])), 0, 255, cv2.NORM_MINMAX,
                             cv2.CV_8U)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmented = np.zeros_like(gray)
        cv2.drawContours(segmented, contours, -1, (255), thickness=cv2.FILLED)
        # removing background
        for x in range(image.shape[1]):
            for y in range(image.shape[0]):
                if segmented[y][x] == 0:
                    bg_mask[y][x][z] = 1
                else:
                    break
        for x in range(image.shape[1] - 1, -1, -1):
            for y in range(image.shape[0] - 1, -1, -1):
                if segmented[y][x] == 0:
                    bg_mask[y][x][z] = 1
                else:
                    break
    heatmap_mask[np.where((heatmap > threshold) & (bg_mask != 1))] = 1
    return heatmap_mask


def get_transformations_transformer(size):
    """
    Get data transformation pipelines.
    Args:
        size (int): size for the input image. Final input shape will be (`size`, `size`, `size`).
    Returns:
        train_transform (monai.transforms.Compose): pipeline for the training input data.
        eval_transform (monai.transforms.Compose): pipeline for the evaluation/testing input data.
    """
    train_transform = Compose([
        LoadImaged(keys='image'),
        EnsureChannelFirstd(keys='image'),
        EnsureTyped(keys='image', dtype=torch.float32),
        Orientationd(keys='image', axcodes='RAS'),
        Spacingd(
            keys='image',
            pixdim=(1.0, 1.0, 1.0),
            mode='bilinear',
            align_corners=True,
            scale_extent=True
        ),
        ScaleIntensityd(keys='image', channel_wise=True),
        CropForegroundd(
            keys='image',
            source_key='image',
            select_fn=(lambda x: x > .3),
            allow_smaller=True
        ),
        Resized(
            keys='image',
            spatial_size=size,
            size_mode='longest',
            mode='bilinear',
            align_corners=True
        ),
        SpatialPadd(keys='image', spatial_size=(size, size, size), mode='minimum'),
        RandAxisFlipd(keys='image', prob=0.5),
        RandRotated(
            keys='image',
            prob=0.5,
            range_x=[.4, .4],
            range_y=[.4, .4],
            range_z=[.4, .4],
            padding_mode='zeros',
            align_corners=True
        ),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
        RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0)
    ])
    return train_transform


# I started with the train test split of colleague and adapt to my task
def train_test_splitting(
        data_folder,
        meta_folder,
        explanation_folder,
        channels,
        features,
        train_ratio=.8,
        multiclass=False,
        verbose=True
):
    """
    Splitting train/eval/test.
    Args:
        data_folder (str): path of the folder containing images.
        meta_folder (str): path of the folder containing csv files.
        explanation_folder (str): path of the folder containing csv files of the explanation.
        channels (list): image channels to select (values `T1w`, `T2w` or both).
        features (list): features set to select.
        train_ratio (float): ratio of the training set, value between 0 and 1.
        multiclass (bool): `False` for binary classification, `True` for ternary classification.
        verbose (bool): whether or not print information.
    Returns:
        train_data (list): the training data ready to feed monai.data.Dataset
        eval_data (list): the evaluation data ready to feed monai.data.Dataset
        test_data (list): the testing data ready to feed monai.data.Dataset.
        (see https://docs.monai.io/en/latest/data.html#monai.data.Dataset).
    """
    scaler = MinMaxScaler()
    df = pd.read_csv(os.path.join(meta_folder, 'data_num.csv'))
    df1 = df[(df['weight'] != .0) & (df['height'] != .0)]
    df['bmi'] = round(df1['weight'] / (df1['height'] * df1['height']), 0)
    df['bmi'] = df['bmi'].fillna(.0)
    sessions = [s.split('_')[0] for s in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, s))]
    subjects = list(set(sessions))

    # uploading of the dataset
    explanation = pd.read_csv(explanation_folder + 'explaination.csv', sep=';')

    # applying splitting on subjects to prevent data leakage
    random.shuffle(subjects)
    split_train = int(len(subjects) * train_ratio)
    train_subjects, test_subjects = subjects[:split_train], subjects[split_train:]
    split_eval = int(len(train_subjects) * .8)
    eval_subjects = train_subjects[split_eval:]
    train_subjects = train_subjects[:split_eval]

    # applying multiclass label correction and splitting
    if multiclass:
        train_subjects, eval_subjects, test_subjects = [], [], []
        df.loc[df['cdr'] == .0, 'final_dx'] = .0
        df.loc[df['cdr'] == .5, 'final_dx'] = 1.
        df.loc[(df['cdr'] != .0) & (df['cdr'] != .5), 'final_dx'] = 2.
        m = np.min(np.unique(df['final_dx'].to_numpy(), return_counts=True)[1])
        df = pd.concat([
            df[df['final_dx'] == .0].sample(m),
            df[df['final_dx'] == 1.].sample(m),
            df[df['final_dx'] == 2.].sample(m)
        ], ignore_index=True)
        n_test = m - int(m * train_ratio)
        n_eval = m - n_test - int(m * train_ratio * train_ratio)
        for i in range(3):
            sub = list(set(df[df['final_dx'] == float(i)]['subject_id'].to_numpy()))
            random.shuffle(sub)
            counter = 0
            for j in range(len(sub)):
                counter += len(df[df['subject_id'] == sub[j]])
                if counter <= n_test:
                    test_subjects.append(sub[j])
                elif counter > n_test and counter <= (n_test + n_eval):
                    eval_subjects.append(sub[j])
                else:
                    train_subjects.append(sub[j])

    # loading sessions paths
    X_train = df[df['subject_id'].isin(train_subjects)]
    X_eval = df[df['subject_id'].isin(eval_subjects)]
    X_test = df[df['subject_id'].isin(test_subjects)]
    train_sessions = [os.path.join(data_folder, s) for s in X_train['session_id'].values]
    eval_sessions = [os.path.join(data_folder, s) for s in X_eval['session_id'].values]
    test_sessions = [os.path.join(data_folder, s) for s in X_test['session_id'].values]

    # loading explanation of subjects
    explanation_train = explanation[explanation['subject_id'].isin(X_train['subject_id'].values)]
    explanation_eval = explanation[explanation['subject_id'].isin(X_eval['subject_id'].values)]
    explanation_test = explanation[explanation['subject_id'].isin(X_test['subject_id'].values)]

    # scaling numerical data in range [0,1]
    X_train.loc[:, features] = scaler.fit_transform(X_train[features])
    X_eval.loc[:, features] = scaler.fit_transform(X_eval[features])
    X_test.loc[:, features] = scaler.fit_transform(X_test[features])

    if platform == 'win32':
        train_data = [dict({
            'image': sorted([os.path.join(s, i) for i in os.listdir(s) if any(c in i for c in channels)]),
            'data': X_train[X_train['session_id'] == s.split('\\')[-1]][features].values[0],
            'label': df[df['session_id'] == s.split('\\')[-1]]['final_dx'].values[0],
            'explanation':
                explanation_train[explanation_train['session_id'] == s.split('\\')[-1]]['explaination'].values[0],
            'session_id': s.split('\\')[-1]
        }) for s in train_sessions]

        eval_data = [dict({
            'image': sorted([os.path.join(s, i) for i in os.listdir(s) if any(c in i for c in channels)]),
            'data': X_eval[X_eval['session_id'] == s.split('\\')[-1]][features].values[0],
            'label': df[df['session_id'] == s.split('\\')[-1]]['final_dx'].values[0],
            'explanation': explanation_eval[explanation_eval['session_id'] == s.split('\\')[-1]]['explaination'].values[
                0],
            'session_id': s.split('\\')[-1]
        }) for s in eval_sessions]

        test_data = [dict({
            'image': sorted([os.path.join(s, i) for i in os.listdir(s) if any(c in i for c in channels)]),
            'data': X_test[X_test['session_id'] == s.split('\\')[-1]][features].values[0],
            'label': df[df['session_id'] == s.split('\\')[-1]]['final_dx'].values[0],
            'explanation': explanation_test[explanation_test['session_id'] == s.split('\\')[-1]]['explaination'].values[
                0],
            'session_id': s.split('\\')[-1]
        }) for s in test_sessions]

    else:
        # arranging data in dictionaries
        # I will also take the reference session of the explanation and the image
        train_data = [dict({
            'image': sorted([os.path.join(s, i) for i in os.listdir(s) if any(c in i for c in channels)]),
            'data': X_train[X_train['session_id'] == s.split('/')[-1]][features].values[0],
            'label': df[df['session_id'] == s.split('/')[-1]]['final_dx'].values[0],
            'explanation':
                explanation_train[explanation_train['session_id'] == s.split('/')[-1]]['explaination'].values[0],
            'session_id': s.split('/')[-1]
        }) for s in train_sessions]
        eval_data = [dict({
            'image': sorted([os.path.join(s, i) for i in os.listdir(s) if any(c in i for c in channels)]),
            'data': X_eval[X_eval['session_id'] == s.split('/')[-1]][features].values[0],
            'label': df[df['session_id'] == s.split('/')[-1]]['final_dx'].values[0],
            'explanation': explanation_eval[explanation_eval['session_id'] == s.split('/')[-1]]['explaination'].values[
                0],
            'session_id': s.split('/')[-1]
        }) for s in eval_sessions]
        test_data = [dict({
            'image': sorted([os.path.join(s, i) for i in os.listdir(s) if any(c in i for c in channels)]),
            'data': X_test[X_test['session_id'] == s.split('/')[-1]][features].values[0],
            'label': df[df['session_id'] == s.split('/')[-1]]['final_dx'].values[0],
            'explanation': explanation_test[explanation_test['session_id'] == s.split('/')[-1]]['explaination'].values[
                0],
            'session_id': s.split('/')[-1]
        }) for s in test_sessions]

    # print data splitting information
    if verbose:
        print(''.join(['> ' for _ in range(40)]))
        print(f'\n{"":<20}{"TRAINING":<20}{"EVALUATION":<20}{"TESTING":<20}\n')
        print(''.join(['> ' for _ in range(40)]))
        tsb1 = str(len(train_subjects)) + ' (' + str(
            round((len(train_subjects) * 100 / len(df['subject_id'].unique())), 0)) + ' %)'
        tsb2 = str(len(eval_subjects)) + ' (' + str(
            round((len(eval_subjects) * 100 / len(df['subject_id'].unique())), 0)) + ' %)'
        tsb3 = str(len(test_subjects)) + ' (' + str(
            round((len(test_subjects) * 100 / len(df['subject_id'].unique())), 0)) + ' %)'
        tss1 = str(len(train_sessions)) + ' (' + str(round((len(train_sessions) * 100 / len(df)), 2)) + ' %)'
        tss2 = str(len(eval_sessions)) + ' (' + str(round((len(eval_sessions) * 100 / len(df)), 2)) + ' %)'
        tss3 = str(len(test_sessions)) + ' (' + str(round((len(test_sessions) * 100 / len(df)), 2)) + ' %)'
        print(f'\n{"subjects":<20}{tsb1:<20}{tsb2:<20}{tsb3:<20}\n')
        print(f'{"sessions":<20}{tss1:<20}{tss2:<20}{tss3:<20}\n')

    return train_data, eval_data, test_data


def plot_grad_cam_explanation(image, label, pred, heatmap, mask, caption, alpha=128):
    """
    Plots model input image, Grad-CAM heatmap, segmentation mask and the explanation generated
    Args:
        image (numpy.ndarray): the input 3D image.
        label (int): the input image label.
        pred (int): model prediction for input image.
        heatmap (numpy.ndarray): the Grad-CAM 3D heatmap.
        mask (numpy.ndarray): the computed 3D segmentation mask.
        caption (string): the explanation generated caption.
        alpha (int): transparency channel. Between 0 and 255.
    Returns:
        None.
    """
    if alpha >= 0 and alpha <= 255:
        heatmap_mask = np.zeros((image.shape[0], image.shape[1], image.shape[2], 4), dtype='uint8')
        heatmap_mask[mask == 1] = [255, 0, 0, alpha]
        image = image[:, :, int(image.shape[2] / 2)]
        heatmap = heatmap[:, :, int(heatmap.shape[2] / 2)]
        heatmap_mask = heatmap_mask[:, :, int(heatmap_mask.shape[2] / 2), :]
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        norm_img = cv2.normalize(image, np.zeros((image.shape[1], image.shape[0])), 0, 1, cv2.NORM_MINMAX)
        im_shows = [
            axs[0].imshow(norm_img, cmap='gray', interpolation='bilinear', vmin=.0, vmax=1.),
            axs[1].imshow(heatmap, cmap='jet', interpolation='bilinear', vmin=.0, vmax=1.),
            axs[2].imshow(norm_img, cmap='gray', interpolation='bilinear', vmin=.0, vmax=1.)
        ]
        axs[2].imshow(heatmap_mask, interpolation='bilinear')
        axs[0].set_title(
            'Label=' + ('NON-AD' if label == 0 else 'AD') + ' | Prediction=' + ('NON-AD' if pred == 0 else 'AD'),
            fontsize=16)
        axs[1].set_title('Grad-CAM Heatmap', fontsize=16)
        axs[2].set_title('Mask - Threshold ' + str(.8), fontsize=16)
        for i, ax in enumerate(axs):
            ax.axis('off')
            fig.colorbar(im_shows[i], ax=ax, ticks=np.linspace(0, 1, 6))

        # insert of caption generated
        caption = "\n".join(textwrap.wrap(caption, width=230))
        fig.text(0.5, -0.3, caption, ha='center',va='center')
        fig.tight_layout()
        fig.suptitle('Alzheimer classification and explaination', fontsize=30, y=1.1)

        plt.show()
    else:
        print('\n' + ''.join(['> ' for i in range(30)]))
        print('\nERROR: alpha channel \033[95m ' + alpha + '\033[0m out of range [0,255].\n')
        print(''.join(['> ' for i in range(30)]) + '\n')


def preprocessing_text(text: str):
    text = text.lower()
    text = text.replace('*','')
    text = text.replace('\n',' ')
    text = text.replace('\t',' ')
    return text
