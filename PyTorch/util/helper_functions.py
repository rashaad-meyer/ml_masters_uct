import os
import zipfile


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


def main():
    download_and_unzip_div2k(downsample='X2')


if __name__ == '__main__':
    main()
