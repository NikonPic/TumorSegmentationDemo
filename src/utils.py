import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
import itertools
from tqdm import tqdm
from sklearn.metrics import accuracy_score, auc
from fastai.metrics import roc_curve
import matplotlib.pyplot as plt
import os
from PIL import Image
import nrrd
from radiomics import featureextractor

if os.getcwd().__contains__('src'):
    from categories import make_categories_advanced, reverse_cat_list
else:
    from .categories import make_categories_advanced, reverse_cat_list

VALID_PART = 0.15
TEST_PART = 0.15
SEED = 53
np.random.seed(SEED)


F_KEY = 'FileName (png)'
CLASS_KEY = 'Aggressiv/Nicht-aggressiv'
ENTITY_KEY = 'Tumor.Entitaet'


def get_advanced_dis_df(df, mode=False):
    """
    redefine the dataframe distribution for advanced training -> separate by entities!
    """
    # 1. get number of entities in overall_df
    # 2. split entities according to train, val / test-split

    # init the empyt idx lists
    train_idx = []
    valid_idx = []
    test_idx = []

    # get the categories by which to split
    cats = reverse_cat_list

    for cat in cats:
        # get all matching df entries
        df_loc = df.loc[df[ENTITY_KEY] == cat]
        loclen = len(df_loc)

        # now split acc to the indices
        validlen = round(loclen * VALID_PART)
        testlen = round(loclen * TEST_PART)
        trainlen = loclen - validlen - testlen

        # get the matching indices and extend the idx list
        df_loc_train = df_loc.iloc[:trainlen]
        train_idx.extend(list(df_loc_train.index))

        df_loc_valid = df_loc.iloc[trainlen:trainlen+validlen]
        valid_idx.extend(list(df_loc_valid.index))

        df_loc_test = df_loc.iloc[trainlen+validlen::]
        test_idx.extend(list(df_loc_test.index))

    # summarize in dictionary
    dis = {
        'train': {
            'len': len(train_idx),
            'idx': train_idx,
        },
        'valid': {
            'len': len(valid_idx),
            'idx': valid_idx,
        },
        'test': {
            'len': len(test_idx),
            'idx': test_idx,
        }
    }

    if mode:
        dis = {
            'test_external': {
                'len': len(df),
                'idx': list(range(len(df))),
            }
        }

    return dis


def calculate_age(born, diag):
    """get the age from the calendar dates"""
    born = datetime.strptime(born, "%d.%m.%Y").date()
    diag = datetime.strptime(diag, "%d.%m.%Y").date()
    return diag.year - born.year - ((diag.month, diag.day) < (born.month, born.day))


def get_acc(interp):
    """get the accuracy of the current interp set, using scipy"""
    return accuracy_score(interp.y_true, interp.pred_class)


def plot_roc_curve(interp, indx=1, lw=2, off=0.02):
    """draw the roc curve"""
    x, y = roc_curve(interp.preds[:, indx], interp.y_true)
    auc_v = auc(x, y)
    plt.figure("roc-curve", figsize=(8, 8))
    plt.plot(x, y, color='darkorange',
             label='ROC curve (area = %0.2f)' % auc_v)
    plt.grid(0.25)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0-off, 1.0])
    plt.ylim([0.0, 1.0 + off])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")


def get_df_paths():
    """collect dataframe and all relevant paths:"""
    # get working directory path
    path = os.getcwd()

    add = "../" if path[-3:] == "src" else ""

    name = 'datainfo.csv'
    pic_folder = 'Images'
    seg_folder = 'Segmentations'

    # get all releevant paths
    paths = {
        "csv": os.path.join(path, f'{add}{name}'),
        "pic": os.path.join(path, f'{add}{pic_folder}'),
        "seg": os.path.join(path, f'{add}{seg_folder}'),
    }

    # get df
    df = pd.read_csv(paths["csv"], header='infer', delimiter=';')

    return df, paths


def get_df_dis(df, born_key='OrTBoard_Patient.GBDAT', diag_key='Erstdiagnosedatum',
               t_key='Tumor.Entitaet', pos_key='Befundlokalisation', out=True,
               mode=False):
    """extract ages and other information from df"""

    # get ages
    if mode:
        ages = df['Alter bei Erstdiagnose']
    else:
        ages = [calculate_age(born, diag) for (born, diag) in zip(
            df[born_key], df[diag_key])]

    # get labels
    labels = [float(lab) for lab in df[CLASS_KEY]]

    # get male(0) / female(1)
    if mode:
        fm = [1 if d_loc == 'f' else 0 for d_loc in df['Geschlecht']]
    else:
        fm = [int(name[0] == 'F') for name in df[F_KEY]]

    # tumor_kind
    tumor_kind = df[t_key]

    # position
    position = df[pos_key]

    # get the shuffled indexes
    dis = get_advanced_dis_df(df, mode=mode)

    if out:
        for key in dis.keys():
            print(f"{key}:")
            print_info(ages, labels, fm, dis[key]['idx'], tumor_kind, position)

        print("All:")
        print_info(ages, labels, fm, list(
            range(len(ages))), tumor_kind, position)

    return ages


def print_info(ages, labels, fm, active_idx, tumor_kind, position, nums=1):
    """summarize all informations as a print message"""

    age = np.array([ages[i] for i in active_idx]).mean().round(nums)
    age_std = np.array([ages[i]
                        for i in active_idx]).std().round(nums)
    print(f'Age: {age} ± {age_std}')

    females = np.array([fm[i] for i in active_idx]).sum()
    femals_p = round((100*females) / len(active_idx), nums)
    print(f'Female: {females} ({femals_p}%)')

    malign = int(np.array([labels[i] for i in active_idx]).sum())
    malign_p = round((100 * malign) / len(active_idx), nums)
    print(f'Malignancy: {malign} ({malign_p}%)')
    print(f'Benign: {len(active_idx)-malign} ({100-malign_p}%)')

    _, cat_mapping = make_categories_advanced(simple=False)

    tumor_list = list(cat_mapping.keys())

    for tumor in tumor_list:
        tums = [int(tumor == name)
                for name in tumor_kind[active_idx]]
        num_tums = np.array(tums).sum()
        per_tum = round(100 * num_tums / len(active_idx), nums)
        print(f'{tumor}: {num_tums} ({per_tum}%)')

    position_dict = {}
    position_dict['Torso/head'] = ['Becken',
                                   'Thoraxwand', 'Huefte', 'LWS', 'os sacrum']
    position_dict['Upper Extremity'] = [
        'Oberarm', 'Hand', 'Schulter', 'Unterarm']
    position_dict['Lower Extremity'] = [
        'Unterschenkel', 'Fuß', 'Knie', 'Oberschenkel']

    for pos_k in position_dict.keys():
        cur_pos = [int(p in position_dict[pos_k])
                   for p in position[active_idx]]
        num_pos = np.array(cur_pos).sum()
        per_pos = round(100 * num_pos / len(active_idx), nums)
        print(f'{pos_k}: {num_pos} ({per_pos}%)')

    dset_part = round(100 * len(active_idx) / len(ages), nums)
    print(f'Dataset Nums: {len(active_idx)} ({dset_part}%)\n\n')


def apply_cat(train, valid, test, dis, new_name, new_cat):
    """add a new category to the dataframe"""
    train_idx = dis['train']['idx']
    valid_idx = dis['valid']['idx']
    test_idx = dis['test']['idx']

    train[new_name] = [new_cat[idx] for idx in train_idx]
    valid[new_name] = [new_cat[idx] for idx in valid_idx]
    test[new_name] = [new_cat[idx] for idx in test_idx]
    return train, valid, test


def png2nrrd(pic_path, nrrd_path):
    """generate nrrd files from png files"""
    for file in tqdm(os.listdir(pic_path)):
        if file.endswith('.png'):
            filename = f'{pic_path}/{file}'
            # task
            img = np.array(Image.open(filename))
            sh = img.shape
            img = img.reshape((sh[2], sh[1], sh[0]))
            # save
            nrrd.write(f'{nrrd_path}/{file[:-4]}.nrrd', img)


def get_radiomics_from_df(df, paths):
    """perform radiomics analysis for all modes"""
    # get the shuffled indexes
    dis = get_advanced_dis_df(df)

    for mode in ['test', 'train', 'valid']:

        # get the active indices
        indices = dis[mode]['idx']

        # make empty coco_dict
        radiomics_extract(df, mode, indices)


def radiomics_extract(df, mode, idxs, path_img='../Images_nrrd', path_seg='../label'):
    """contains the radiomics feature extraction"""

    set_path = './pyradiomics_settings.yaml'
    extractor = featureextractor.RadiomicsFeatureExtractor(set_path)
    df_list = []
    except_dict = {
        'idx': [],
    }

    for idx in tqdm(idxs):
        # get current filename
        o = df.iloc[idx]
        file = o[F_KEY]

        # get the two relevant paths for image and segmentation
        picpath = f'{path_img}/{file}.nrrd'
        nrrdpath = f'{path_seg}/{file}.nrrd'

        print(picpath)
        print(nrrdpath)

        # obtain result
        result = extractor.execute(picpath, nrrdpath)

        # compress
        df_loc = result2compresdf(result)
        # append the label
        df_loc['label1'] = o[CLASS_KEY]
        df_loc['label2'] = o[ENTITY_KEY]

        df_list.append(df_loc.copy())

    df_rad = df_list[0]
    df_except = pd.DataFrame.from_dict(except_dict, orient='index')

    for i, df_loc in enumerate(df_list):
        if i > 0:
            df_rad = df_rad.append(df_loc)

    # finnaly save result
    df_rad.to_csv(f'../radiomics/{mode}.csv')
    df_except.to_csv(f'../radiomics/{mode}-except.csv')


def result2compresdf(result: dict):
    """compress the result to maintain the relevant features only"""
    comp_res = {}

    for key in result.keys():
        val = result[key]

        if type(val) in [np.ndarray]:
            comp_res[key] = float(val)

    df = pd.DataFrame.from_dict(comp_res, orient='index')

    return df.T

def plot_confusion_matrix(
    conf_mat, target_names, title="Confusion matrix", cmap=None, normalize=False, font=16
):
    """
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    accuracy = np.trace(conf_mat) / float(np.sum(conf_mat))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    fig = plt.figure(figsize=(16, 16))
    plt.imshow(conf_mat, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=font)
    # plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90, fontsize=font)
        plt.yticks(tick_marks, target_names, fontsize=font)

    if normalize:
        conf_mat = conf_mat.astype(
            "float") / conf_mat.sum(axis=1)[:, np.newaxis]

    thresh = conf_mat.max() / 1.5 if normalize else conf_mat.max() / 2
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        if normalize:
            plt.text(
                j,
                i,
                "{:0.4f}".format(conf_mat[i, j]),
                horizontalalignment="center",
                color="white" if conf_mat[i, j] > thresh else "black", fontsize=font
            )
        else:
            plt.text(
                j,
                i,
                "{:,}".format(conf_mat[i, j]),
                horizontalalignment="center",
                color="white" if conf_mat[i, j] > thresh else "black", fontsize=font
            )

    # plt.tight_layout()
    plt.ylabel("Histopathology as standard of reference", fontsize=font)
    plt.xlabel(
        "Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(
            accuracy, misclass), fontsize=font
    )
    plt.show()
    return fig
