import os
import configparser
import shutil
import pandas as pd
import numpy as np

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

df_mw_download = pd.read_csv(os.path.join(BASE_PATH, 'mw_2016_download.csv'))
df_mw_download_info = pd.read_csv(os.path.join(BASE_PATH, 'mw_2016_download_info.csv'))
df_mw_guide = pd.read_csv(os.path.join(BASE_PATH, 'mw_2016_guide.csv'))

if len(df_mw_download) != len(df_mw_download_info):
    # this means the script broke and you restarted it
    # this is how I named images so it should work...
    df_mw_download['images'] = df_mw_download.apply(lambda x: str(x.im_lat) + '_' + str(x.im_lon) + '.png', axis=1)
else:
    df_mw_download = df_mw_download_info

# we didn't download duplicate images which explains the difference
df_mw_download.shape, df_mw_guide.shape

len(df_mw_guide.groupby(['clust_lat', 'clust_lon'])) # number of clusters

# merges the images downloaded with the original dataframe
df_sub = df_mw_download[['im_lat', 'im_lon', 'images']]
df_mw = pd.merge(left=df_mw_guide, right=df_sub, on=['im_lat', 'im_lon'])

df_mw.head()

# I didn't download all the images and also I got rid of some image repeats, hence the difference
df_mw_guide.shape, df_mw.shape

# adds a "cluster number" to the dataframe
clust_group = df_mw.groupby(['clust_lat', 'clust_lon'])
clust_group = clust_group.first().reset_index()[['clust_lat', 'clust_lon']]
clust_numbers = np.arange(len(clust_group))
clust_group['clust_num'] = clust_numbers

# print(clust_group.head())
df_mw = pd.merge(df_mw, clust_group, on=['clust_lat', 'clust_lon'])

# print(df_mw.head())
# print(df_mw.shape)
#need to drop NaN in images column to avoid error
df_mw = df_mw[pd.notnull(df_mw['images'])]

df_mw['images_renamed'] = df_mw.apply(lambda x: x.images[:-4] + '_' + str(x.clust_num) + '.png', axis=1)

df_mw.head()

# you can add other countries here under the structure images/<COUNTRY>/
path = os.path.join(BASE_PATH, '..', 'images/ims_mw/ims')
if not os.path.exists(path):
    os.makedirs(path)

def create_im_renamed(x):
    """
    # this will copy images into a folder called ims
    # this folder is helpful because the original folder has all unique images
    # now, we need to duplicate those images and distinguish them by their new
    # name (as made previously)

    shutil.copy(source, destination)

    """
    path = os.path.join(BASE_PATH, '..', '/images/ims_mw/ims/{}'.format(x.images_renamed))

    if not os.path.exists(path):
        shutil.copy(
            os.path.join(BASE_PATH, 'ims_malawi_2016/{}'.format(x.images)),
            os.path.join(BASE_PATH, '..', 'images/ims_mw/ims/{}'.format(x.images_renamed)))

print('Working on moving/renaming files')
df_mw.apply(create_im_renamed, axis=1)
print('Completed file renaming process')

df_mw.head()

df_mw.to_csv(os.path.join(BASE_PATH, 'mw_full_guide.csv'), index=False)

pic_list = df_mw['images'].values.tolist()
to_pick = int(0.8*len(pic_list)); to_pick

inds = np.arange(len(pic_list))
train_ind = np.random.choice(np.arange(len(pic_list)), to_pick, replace=False)
valid_ind = np.delete(inds, train_ind)

pic_list = np.array(pic_list)
train_im = pic_list[train_ind]
valid_im = pic_list[valid_ind]

path_train = os.path.join(BASE_PATH, '..', 'images/ims_mw/train')
if not os.path.exists(path_train):
    os.makedirs(path_train)
path_valid = os.path.join(BASE_PATH, '..', 'images/ims_mw/valid')
if not os.path.exists(path_valid):
    os.makedirs(path_valid)

t = df_mw.iloc[train_ind]
v = df_mw.iloc[valid_ind]

for fi, l in zip(t['images_renamed'], t['nightlight_bin']):
    path = os.path.join(path_valid, '{}'.format(l))
    if not os.path.exists(path):
        os.makedirs(path)
    path_file = os.path.join(BASE_PATH, '..', 'images/ims_mw/train/{}'.format(l))
    if not os.path.exists(path_file):
        source = os.path.join(BASE_PATH, '..', 'images/ims_mw/ims/{}'.format(fi))
        shutil.copy(source, path_file)

for fi, l in zip(v['images_renamed'], v['nightlight_bin']):
    path = os.path.join(path_valid, '{}'.format(l))
    if not os.path.exists(path):
        os.makedirs(path)
    path_file = os.path.join(BASE_PATH, '..', 'images/ims_mw/train/{}'.format(l))
    if not os.path.exists(path_file):
        source = os.path.join(BASE_PATH, '..', 'images/ims_mw/ims/{}'.format(fi))
        shutil.copy(source, path_file)

# shows count distribution in train folder
for i in range(1,4):
    path = os.path.join(BASE_PATH, '..', 'images/ims_mw/train/{}'.format(str(i)))
    num = len(os.listdir(path))
    print('Train folder {} contains {} files'.format(i, num))

# shows count distribution in valid folder
for i in range(1,4):
    path = os.path.join(BASE_PATH, '..', 'images/ims_mw/valid/{}'.format(str(i)))
    num = len(os.listdir(path))
    print('Valid folder {} contains {} files'.format(i, num))
