from pathlib import Path
import wget
import zipfile


images = Path('datasets/South-Building/images/')

if not images.exists():
    wget.download(
        'http://cvg.ethz.ch/research/local-feature-evaluation/South-Building.zip',
        out = 'datasets/'
    )
    with zipfile.ZipFile('datasets/South-Building.zip', 'r') as zip_ref:
        zip_ref.extractall('datasets/')
