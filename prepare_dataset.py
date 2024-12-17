import json
import pdb
from tqdm import tqdm
import os
import shutil
from PIL import Image
import glob


def copy_files_w_resize(class_info, split_name):
    src_basedir = './김치/'
    dst_basedir = '../dataset/'

    img_ids = None
    if split_name == 'train':
        img_ids = range(0, 700)
    elif split_name == 'val':
        img_ids = range(700, 800)
    elif split_name == 'test':
        img_ids = range(800, 1000)

    img_size = (64, 64)

    # create class sub dir
    src_dir_name = os.path.join(src_basedir, class_info['ko'])
    dst_dir_name = os.path.join(dst_basedir, split_name, class_info['en'])
    os.makedirs(dst_dir_name, exist_ok=True)
    # copy files
    for img_id in tqdm(img_ids, class_info['en']+' '+split_name):
        src_fn = glob.glob(os.path.join(src_dir_name, class_info['fn_prefix'] + '_' + str(img_id).zfill(4) + '.*'))[0]
        dst_fn = os.path.join(dst_dir_name, class_info['fn_prefix'] + '_' + str(img_id).zfill(4) + '.jpg')
        # resize and save
        img = Image.open(src_fn)
        img_resized = img.resize(img_size)
        img_resized.convert('RGB').save(dst_fn)





class_names_en = ['got', 'kkakdoogi', 'nabak', 'moosaengchae', 'baechu', 'baik', 'boochoo', 'yeolmoo', 'ohyeesobaki', 'chongkak', 'pa']

class_names_kr = ['갓김치', '깍두기', '나박김치', '무생채', '배추김치', '백김치', '부추김치', '열무김치', '오이소박이', '총각김치', '파김치']

fn_prefix = ['Img_029', 'Img_030', 'Img_031', 'Img_032', 'Img_033', 'Img_034', 'Img_035', 'Img_036', 'Img_037', 'Img_038', 'Img_039']

class_infos = []
for i, _ in enumerate(class_names_en):
    class_infos.append({'en': class_names_en[i], 'ko': class_names_kr[i], 'fn_prefix': fn_prefix[i]})

for class_info in class_infos:
    os.makedirs('./dataset/train/'+class_info['en'], exist_ok=True)
    os.makedirs('./dataset/val/'+class_info['en'], exist_ok=True)
    os.makedirs('./dataset/test/'+class_info['en'], exist_ok=True)


print('preparing the dataset...')
# for training set
# unzip
os.chdir('./한국 음식 이미지/')
os.system('tar -xvf kfood.zip')
os.system('rm -rf 구이.zip 국.zip 기타.zip 나물.zip 떡.zip 만두.zip 면.zip 무침.zip 밥.zip 볶음.zip 쌈.zip 음청류.zip 장.zip 장아찌.zip 적.zip 전.zip 전골.zip 조림.zip 죽.zip 찌개.zip 찜.zip 탕.zip 튀김.zip 한과.zip 해물.zip 회.zip')
os.system('tar -xvf 김치.zip')

for i, class_info in enumerate(class_infos):
    copy_files_w_resize(class_info, 'train')
    copy_files_w_resize(class_info, 'val')
    copy_files_w_resize(class_info, 'test')

print('ready to use the dataset!')