import csv
import os
import re
import zipfile
import tarfile
import requests
import pandas as pd
from datetime import datetime
import xml.etree.ElementTree as ET
import shutil


def unzip_data(filename, path):
    if filename.endswith("zip"):
        zip_ref = zipfile.ZipFile(filename, "r")
        zip_ref.extractall(path)
        zip_ref.close()
    elif filename.endswith("tar.gz"):
        tar = tarfile.open(filename, "r:gz")
        tar.extractall(path)
        tar.close()
    elif filename.endswith("tar"):
        with tarfile.open(filename, 'r') as tar:
            # Extract all files from the tar file
            tar.extractall(path)
    os.remove(filename)


def download_div2k(path, downsample, dataset_type='train'):
    if dataset_type not in ['train', 'valid']:
        raise ValueError("dataset_type should be either 'train' or 'valid'")

    hr_url = f'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_{dataset_type}_HR.zip'
    lr_url = f"https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_{dataset_type}_LR_bicubic_{downsample}.zip"

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Create new path: {path}")

    lr_filename = f"{path}/DIV2K_{dataset_type}_LR_bicubic_{downsample}.zip"
    hr_filename = f"{path}/DIV2K_{dataset_type}_HR.zip"

    if not os.path.exists(lr_filename):
        print(f'Downloading {lr_filename}...')
        os.system(f"curl {lr_url} --output {lr_filename}")
    else:
        print(f'{lr_filename} detected')

    if not os.path.exists(hr_filename):
        print(f'Downloading {hr_filename}')
        os.system(f"curl {hr_url} --output {hr_filename}")
    else:
        print(f'{hr_filename} detected')

    return lr_filename, hr_filename


def download_and_unzip_div2k(path='data', downsample='X2', dataset_type='train'):
    lr_path = f'{path}/DIV2K_{dataset_type}_LR_bicubic/{downsample}'
    hr_path = f'{path}/DIV2K_{dataset_type}_HR'

    if os.path.exists(lr_path) and os.path.exists(hr_path):
        print(f'{dataset_type.upper()} HR and LR data are both downloaded and unzipped already!')
        return lr_path, hr_path
    else:
        lr_filename, hr_filename = download_div2k(path, downsample, dataset_type)

    if not os.path.exists(lr_path):
        print(f'Unzipping {dataset_type.upper()} LR zip...')
        unzip_data(lr_filename, path)
    else:
        print(f'{dataset_type.upper()} LR already unzipped!')

    if not os.path.exists(hr_path):
        print(f'Unzipping {dataset_type.upper()} HR zip...')
        unzip_data(hr_filename, path)
    else:
        print(f'{dataset_type.upper()} HR already unzipped!')

    return lr_path, hr_path


def download_and_unzip_voc_ds(path='data/obj-det'):
    links = [
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    ]
    file_names = [link.split('/')[-1][:-4] for link in links]

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Create new path: {path}")

    for link, file_name in zip(links, file_names):
        file_path = f'{path}/{file_name}'
        print(f'Downloading {file_name}...')
        download_tar(link, f"{file_path}.tar")
        print(f'Unzipping {file_name}')
        unzip_data(f"{file_path}.tar", path)


def download_div2k_h5(output_file):
    import gdown
    file_name = output_file.split('/')[-1]

    # check if path exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created new path: {output_dir}")
    elif os.path.exists(output_file):
        print('Div2k file already downloaded!')
        return

    download_ids = {

        'div2k_patches_x2_ycbcr_32.h5': '1CFCAE5XSIfViZx6YiqKRCgbGigp3-GOg',
        'div2k_x2_ycbcr_32.h5': '1t_1wuZOaEF1SA6ppbnISVopki7XD5LU_',
    }
    try:
        download_url = f'https://drive.google.com/uc?id={download_ids[file_name]}'
    except:
        raise ValueError(f'Please select a output file name {download_ids.keys()}')
    gdown.download(download_url, output_file, quiet=False)


def download_and_unzip_sr_ds(path='data/sr', ds_name='Set5'):
    links = {
        'Set5': 'https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip',
        'Set14': 'https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip',
        'BSD100': 'https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip'
    }
    link = links[ds_name]
    os.makedirs(path, exist_ok=True)
    file_path = f'{path}/{ds_name}'

    if not os.path.exists(file_path):
        print(f'Downloading {ds_name}...')
        download_tar(link, f"{file_path}.zip")

        print(f'Unzipping {file_path}.zip')
        if ds_name == 'BSD100':
            unzip_data(f"{file_path}.zip", file_path)
        else:
            unzip_data(f"{file_path}.zip", path)

    else:
        print(f'Files found for {ds_name}')
    return file_path


def download_91_image_and_set5_ds(path='data/sr/srcnn', scale=3):
    download_urls = {
        '91-image': [
            'https://www.dropbox.com/s/2hsah93sxgegsry/91-image_x2.h5?dl=1',
            'https://www.dropbox.com/s/curldmdf11iqakd/91-image_x3.h5?dl=1',
            'https://www.dropbox.com/s/22afykv4amfxeio/91-image_x4.h5?dl=1'
        ],
        'Set5': [
            'https://www.dropbox.com/s/r8qs6tp395hgh8g/Set5_x2.h5?dl=1',
            'https://www.dropbox.com/s/58ywjac4te3kbqq/Set5_x3.h5?dl=1',
            'https://www.dropbox.com/s/0rz86yn3nnrodlb/Set5_x4.h5?dl=1'
        ],

    }

    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    output_files = {}

    for dataset, urls in download_urls.items():
        url = urls[scale - 2]
        file_name = url.split('/')[-1].split('?')[0]  # Extract the file name from the URL
        output_files[dataset] = os.path.join(path, file_name)
        command = f"curl -L -o {output_files[dataset]} {url}"
        if not os.path.exists(output_files[dataset]):
            os.system(command)
        else:
            print(f'{output_files[dataset]} already downloaded')

    print("Download completed.")
    return output_files


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(path, year, image_id, classes):
    in_file = open('%s/VOCdevkit/VOC%s/Annotations/%s.xml' % (path, year, image_id))
    out_file = open('%s/VOCdevkit/VOC%s/labels/%s.txt' % (path, year, image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def get_voc_labels_from_ds(path='data/obj-det'):
    sets = [('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    for year, image_set in sets:
        if not os.path.exists(f'%s/VOCdevkit/VOC%s/labels/' % (path, year)):
            os.makedirs('%s/VOCdevkit/VOC%s/labels/' % (path, year))
        image_ids = open('%s/VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (path, year, image_set)).read().strip().split()
        list_file = open('%s/%s_%s.txt' % (path, year, image_set), 'w')
        for image_id in image_ids:
            list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n' % (path, year, image_id))
            convert_annotation(path, year, image_id, classes)
        list_file.close()


def download_tar(url, target_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())


def is_obj_det_ds_downloaded(BASE_DIR):
    obj_det_files = [
        'images',
        'labels',
        'train.csv',
        'test.csv',
    ]
    files_in_dir = os.listdir(BASE_DIR)

    for file in obj_det_files:
        if file not in files_in_dir:
            return False

    return True


def clean_voc_ds_text_files(base_dir='data/obj-det'):
    get_voc_labels_from_ds()

    # Concatenate the text files into train.txt
    with open(os.path.join(base_dir, "train.txt"), "w") as output_file:
        txt_files = ["2007_train.txt", "2007_val.txt"]
        txt_files += [f for f in os.listdir(base_dir) if f.startswith("2012_") and f.endswith(".txt")]

        for file_name in txt_files:
            with open(os.path.join(base_dir, file_name)) as f:
                shutil.copyfileobj(f, output_file)

    # Copy 2007_test.txt to test.txt
    shutil.copy(os.path.join(base_dir, "2007_test.txt"), os.path.join(base_dir, "test.txt"))

    # Move txt files we won't be using to clean up a little bit
    old_txt_files_dir = os.path.join(base_dir, "old_txt_files")
    os.makedirs(old_txt_files_dir, exist_ok=True)

    for file_name in os.listdir(base_dir):
        if file_name.startswith("2007") or file_name.startswith("2012"):
            shutil.move(os.path.join(base_dir, file_name), old_txt_files_dir)

    # Execute the generate_csv function
    generate_csv(base_dir)

    # Create directories
    os.makedirs(os.path.join(base_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels"), exist_ok=True)

    # Copy images and labels
    vocdevkit_dir = os.path.join(base_dir, "VOCdevkit")

    for file_name in os.listdir(vocdevkit_dir):
        if file_name.endswith(".jpg"):
            shutil.copy(os.path.join(vocdevkit_dir, file_name), os.path.join(base_dir, "images"))

    for subdir, _, files in os.walk(os.path.join(vocdevkit_dir, "VOC2007/labels")):
        for file_name in files:
            shutil.copy(os.path.join(subdir, file_name), os.path.join(base_dir, "labels"))

    for subdir, _, files in os.walk(os.path.join(vocdevkit_dir, "VOC2012/labels")):
        for file_name in files:
            shutil.copy(os.path.join(subdir, file_name), os.path.join(base_dir, "labels"))

    # Move images and labels
    for subdir, _, files in os.walk(os.path.join(vocdevkit_dir, "VOC2007/JPEGImages")):
        for file_name in files:
            shutil.move(os.path.join(subdir, file_name), os.path.join(base_dir, "images"))

    for subdir, _, files in os.walk(os.path.join(vocdevkit_dir, "VOC2012/JPEGImages")):
        for file_name in files:
            shutil.move(os.path.join(subdir, file_name), os.path.join(base_dir, "images"))

    # Remove VOCdevkit folder
    shutil.rmtree(vocdevkit_dir)

    # Move test.txt and train.txt to old_txt_files
    shutil.move(os.path.join(base_dir, "test.txt"), old_txt_files_dir)
    shutil.move(os.path.join(base_dir, "train.txt"), old_txt_files_dir)


def generate_csv(base_dir):
    read_train = open(f"{base_dir}/train.txt", "r").readlines()

    with open(f"{base_dir}/train.csv", mode="w", newline="") as train_file:
        for line in read_train:
            image_file = line.split("/")[-1].replace("\n", "")
            text_file = image_file.replace(".jpg", ".txt")
            data = [image_file, text_file]
            writer = csv.writer(train_file)
            writer.writerow(data)

    read_train = open(f"{base_dir}/test.txt", "r").readlines()

    with open(f"{base_dir}/test.csv", mode="w", newline="") as train_file:
        for line in read_train:
            image_file = line.split("/")[-1].replace("\n", "")
            text_file = image_file.replace(".jpg", ".txt")
            data = [image_file, text_file]
            writer = csv.writer(train_file)
            writer.writerow(data)


def write_history_to_csv(path, history: dict, model_name, deconv, loss):
    if deconv:
        deconv = 'deconv'
    else:
        deconv = 'conv'

    results_folder = f'{path}/results'

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    now = datetime.now()
    dt_string = now.strftime("%m-%d_%H-%M")
    file_name = f'{dt_string}_{deconv}_{model_name}_{loss}'
    df = pd.DataFrame(history)
    output_filename = f'{results_folder}/{file_name}.csv'
    df.to_csv(output_filename)
    print(f'Results written to {output_filename}')


def write_history_to_csv_by_experiment_name(path, history: dict, experiment_name):
    results_folder = f'{path}/results'

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    now = datetime.now()
    dt_string = now.strftime("%m-%d_%H-%M")
    file_name = f'{dt_string}_{experiment_name}'
    df = pd.DataFrame(history)
    output_filename = f'{results_folder}/{file_name}.csv'
    df.to_csv(output_filename)
    print(f'Results written to {output_filename}')


def get_voc_ds(base_dir='data/obj-det'):
    os.makedirs(base_dir, exist_ok=True)

    if not is_obj_det_ds_downloaded(base_dir):
        download_and_unzip_voc_ds(base_dir)
        clean_voc_ds_text_files(base_dir)
        print('Dataset files downloaded and unzipped')
    else:
        print('Dataset files already downloaded')


def main():
    download_and_unzip_sr_ds(ds_name='set5')
    download_and_unzip_sr_ds(ds_name='set14')
    download_and_unzip_sr_ds(ds_name='bsd100')
    return


if __name__ == '__main__':
    output_file = '../../data/test/div2k_patches_x2_ycbcr_32.h5'
    download_div2k_h5(output_file)
