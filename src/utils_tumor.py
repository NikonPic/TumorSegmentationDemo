import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
from tqdm import tqdm
from sklearn.metrics import accuracy_score, auc
from fastai.metrics import roc_curve
import matplotlib.pyplot as plt
import os

if os.getcwd().__contains__('src'):
    from categories import make_categories_advanced
else:
    from .categories import make_categories_advanced

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
    cats = set(df[ENTITY_KEY])

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
    """
    get the accuracy of the current interp set, using scipy
    """
    return accuracy_score(interp.y_true, interp.pred_class)


def plot_roc_curve(interp, indx=1, lw=2, off=0.02):
    """
    draw the roc curve
    """
    x, y = roc_curve(interp.preds[:, indx], interp.y_true)
    auc_v = auc(x, y)
    plt.figure("roc-curve")
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


def dis_df(df):
    """
    Distribute the dataframe into:
    - Training
    - Validation
    - Test
    """
    # get fixed randomness
    np.random.seed(SEED)
    # get the shuffled indexes
    len_all = len(df)
    idx_list = np.array(list(range(len_all)))
    np.random.shuffle(idx_list)

    # calculate the lengths
    valid_len = int(len_all * VALID_PART)
    test_len = int(len_all * TEST_PART)
    testval_len = valid_len + test_len
    train_len = len_all - testval_len

    # get the active indexes for each dataset
    train_idx = range(testval_len, len_all)
    valid_idx = range(test_len, testval_len)
    test_idx = range(test_len)

    # summarize in dictionary
    dis = {
        'train': {
            'len': train_len,
            'idx': idx_list[train_idx]
        },
        'valid': {
            'len': valid_len,
            'idx': idx_list[valid_idx]
        },
        'test': {
            'len': test_len,
            'idx': idx_list[test_idx]
        }
    }
    return dis


def get_df_paths():
    """
    collect dataframe and all relevant paths:
    """
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
    """
    extract ages and other information from df
    """

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
    """
    summarize all informations as a print message
    """

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
