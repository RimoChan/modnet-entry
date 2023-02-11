import os
import setuptools
import subprocess
import urllib.request


here = os.path.dirname(os.path.abspath(__file__))
entry_folder = os.path.join(here, 'MODNet_entry')
MODNet_folder = os.path.join(entry_folder, 'MODNet')


def run(s):
    subprocess.run(s, check=True)


def download(url: str, path: str):
    f = urllib.request.urlopen(url)
    data = f.read()
    with open(path, 'wb') as f:
        f.write(data)


models = {
    '1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz': 'modnet_photographic_portrait_matting.ckpt',
    '1Nf1ZxeJZJL8Qx9KadcYYyEmmlKhTADxX': 'modnet_webcam_portrait_matting.ckpt',
    # '1gNJXQPUBBp2mbA4q1Giz5mzv3EpxR7lq': 'mobilenetv2_human_seg.ckpt',
    # '1IxxExwrUe4_yQnlEx389tmQI8luX7z5m': 'modnet_photographic_portrait_matting.onnx',
}


if not os.path.exists(MODNet_folder):
    run(['git', 'clone', '--depth', '1', 'https://github.com/ZHKKKe/MODNet.git', MODNet_folder])


for google_id, name in models.items():
    model_path = os.path.join(MODNet_folder, 'pretrained', name)
    if not os.path.exists(model_path):
        download(f'https://drive.google.com/uc?export=download&id={google_id}', model_path)


setuptools.setup(
    name='MODNet_entry',
    version='1.0.0',
    author='RimoChan',
    author_email='the@librian.net',
    description='librian',
    long_description=open('readme.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/RimoChan/rimo-rss-reader',
    packages=[
        'MODNet_entry',
    ],
    package_data={
        'MODNet_entry': ['*', '*/*', '*/*/*', '*/*/*/*', '*/*/*/*/*', '*/*/*/*/*/*']
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=open('requirements.txt', encoding='utf8').read().splitlines(),
    python_requires='>=3.7',
)
