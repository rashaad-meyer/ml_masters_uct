import os
import zipfile
from datetime import datetime
import pandas as pd


def unzip_data(filename, path):
    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall(path)
    zip_ref.close()
    os.remove(filename)


def download_div2k(path, downsample):
    hr_url = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip'
    lr_url = f"https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_{downsample}.zip"

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Create new path: {path}")

    lr_filename = f"{path}/DIV2K_train_LR_bicubic_{downsample}.zip"
    hr_filename = f"{path}/DIV2K_train_HR.zip"

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


def download_and_unzip_div2k(path='data', downsample='X2'):
    lr_path = f'{path}/DIV2K_train_LR_bicubic/{downsample}'
    hr_path = f'{path}/DIV2K_train_HR'

    if os.path.exists(lr_path) and os.path.exists(hr_path):
        print('HR and LR data are both downloaded and unzipped already!')
        return lr_path, hr_path
    else:
        lr_filename, hr_filename = download_div2k(path, downsample)

    if not os.path.exists(lr_path):
        print('Unzipping LR zip...')
        unzip_data(lr_filename, path)
    else:
        print('LR already unzipped!')

    if not os.path.exists(hr_path):
        print('Unzipping HR zip...')
        unzip_data(hr_filename, path)
    else:
        print('HR already unzipped!')

    return lr_path, hr_path


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


def main():
    download_and_unzip_div2k(downsample='X2')


if __name__ == '__main__':
    main()
