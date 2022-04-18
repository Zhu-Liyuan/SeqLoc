import argparse
import shutil
from typing import Optional, List
import multiprocessing
from pathlib import Path
import pycolmap

from . import logger
from .utils.database import COLMAPDatabase, array_to_blob
from .triangulation import (
    import_features, import_matches, geometric_verification, OutputCapture)
import numpy as np
from hloc.utils import parsers


def create_empty_db(database_path):
    if database_path.exists():
        logger.warning('The database already exists, deleting it.')
        database_path.unlink()
    logger.info('Creating an empty database...')
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()


def import_images(image_dir, database_path, camera_mode, image_list=None):
    logger.info('Importing images into the database...')
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f'No images found in {image_dir}.')
    with pycolmap.ostream():
        pycolmap.import_images(database_path, image_dir, camera_mode,
                               image_list=image_list or [])

def import_intrinsics(database_path, intrinsics_path):
    database_path = Path(database_path)
    intrinsics_path = Path(intrinsics_path)
    all_query_images = parsers.parse_image_list(intrinsics_path, with_intrinsics=True)
    db = COLMAPDatabase.connect(database_path)
    q_images = db.execute("SELECT * FROM images")
    q_cameras = db.execute("SELECT * FROM cameras")
    for q_image in q_images:
        for img in all_query_images:
            if q_image[1] in img[0]:
                param = db.execute((f"SELECT * FROM cameras WHERE camera_id={q_image[0]}"))
                camera_id, model, width, height, _, _ = next(param)
                new_params = np.asarray(img[1].params, dtype=np.float64)
                db.execute(f"DELETE FROM cameras WHERE camera_id={q_image[0]}")
                # db.execute(f"UPDATE cameras SET params={new_blob} WHERE camera_id={q_image[0]}")
                db.execute( "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
                        (camera_id, model, width, height, array_to_blob(new_params),
                        0))
    db.commit()
    db.close()

def get_image_ids(database_path):
    db = COLMAPDatabase.connect(database_path)
    images = {}
    for name, image_id in db.execute("SELECT name, image_id FROM images;"):
        images[name] = image_id
    db.close()
    return images


def run_reconstruction(sfm_dir, database_path, image_dir, verbose=False):
    models_path = sfm_dir / 'models'
    models_path.mkdir(exist_ok=True, parents=True)
    logger.info('Running 3D reconstruction...')
    with OutputCapture(verbose):
        with pycolmap.ostream():
            reconstructions = pycolmap.incremental_mapping(
                database_path, image_dir, models_path,
                num_threads=min(multiprocessing.cpu_count(), 16))

    if len(reconstructions) == 0:
        logger.error('Could not reconstruct any model!')
        return None
    logger.info(f'Reconstructed {len(reconstructions)} model(s).')

    largest_index = None
    largest_num_images = 0
    for index, rec in reconstructions.items():
        num_images = rec.num_reg_images()
        if num_images > largest_num_images:
            largest_index = index
            largest_num_images = num_images
    assert largest_index is not None
    logger.info(f'Largest model is #{largest_index} '
                f'with {largest_num_images} images.')

    for filename in ['images.bin', 'cameras.bin', 'points3D.bin']:
        if (sfm_dir / filename).exists():
            (sfm_dir / filename).unlink()
        shutil.move(
            str(models_path / str(largest_index) / filename), str(sfm_dir))
    return reconstructions[largest_index]


def main(sfm_dir, image_dir, pairs, features, matches,
         camera_mode=pycolmap.CameraMode.AUTO, verbose=False,
         skip_geometric_verification=False, min_match_score=None,
         image_list: Optional[List[str]] = None, intrinsics_path=None):

    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / 'database.db'

    create_empty_db(database)
    import_images(image_dir, database, camera_mode, image_list)
    image_ids = get_image_ids(database)
    import_features(image_ids, database, features)
    import_matches(image_ids, database, pairs, matches,
                   min_match_score, skip_geometric_verification)
    import_intrinsics(database, intrinsics_path=intrinsics_path)
    if not skip_geometric_verification:
        geometric_verification(database, pairs, verbose)
    reconstruction = run_reconstruction(sfm_dir, database, image_dir, verbose)
    if reconstruction is not None:
        logger.info(f'Reconstruction statistics:\n{reconstruction.summary()}'
                    + f'\n\tnum_input_images = {len(image_ids)}')
    return reconstruction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--image_dir', type=Path, required=True)

    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)

    parser.add_argument('--camera_mode', type=str, default="AUTO",
                        choices=list(pycolmap.CameraMode.__members__.keys()))
    parser.add_argument('--skip_geometric_verification', action='store_true')
    parser.add_argument('--min_match_score', type=float)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(**args.__dict__)
