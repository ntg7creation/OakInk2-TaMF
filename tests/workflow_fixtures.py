"""Small deterministic original-format caches; no downloaded assets needed."""
import numpy as np

from oakink2_tamf.dataset.cache_util import FIELDS, OBJECTS


def tiny_cache(keys=("seq_a", "seq_b")):
    cache = {key: [] for key in (*FIELDS, OBJECTS)}
    objects = set()
    for index, key in enumerate(keys):
        length = 20 + index
        pose = np.zeros((160, 16, 3, 3), np.float32)
        pose[:length] = np.eye(3, dtype=np.float32)
        # Rotations vary over time, exercising reversal after 6D conversion.
        angle = np.linspace(0, 0.3 + index * 0.1, length)
        pose[:length, 4, 0, 0] = np.cos(angle)
        pose[:length, 4, 1, 1] = np.cos(angle)
        pose[:length, 4, 0, 1] = -np.sin(angle)
        pose[:length, 4, 1, 0] = np.sin(angle)
        tsl = np.zeros((160, 3), np.float32)
        tsl[:length, 0] = np.linspace(0, 1, length)
        shape = np.zeros((160, 10), np.float32)
        shape[:length] = index + 0.25
        traj = {}
        for oid in ("obj_a", "obj_b")[:index % 2 + 1]:
            arr = np.zeros((160, 4, 4), np.float32)
            arr[:length] = np.eye(4, dtype=np.float32)
            arr[:length, 0, 3] = np.linspace(0.5, 0.8, length)
            traj[oid] = arr
        objects.update(traj)
        side = "rh" if index % 2 == 0 else "lh"
        values = ((key, "primitive", side), length, pose, tsl, shape, side,
                  "hold the object", traj, list(range(0, length * 12, 12)))
        for field, value in zip(FIELDS, values):
            cache[field].append(value)
    cache[OBJECTS] = sorted(objects)
    return cache


def configuration(keys=("seq_a", "seq_b")):
    return dict(data=dict(data_prefix="not-needed", obj_embedding_prefix=None,
                          obj_pointcloud_prefix=None, enable_obj_model=False,
                          append_reverse_segment=False),
                train=dict(process_range=list(keys), cache_dict_filepath=None,
                           require_verified_cache=False, batch_size=1),
                val=dict(process_range=["val_seq"], cache_dict_filepath=None, val_freq=-1, batch_size=1),
                test=dict(process_range=["test_seq"], cache_dict_filepath=None, test_freq=-1, batch_size=1),
                runtime=dict(num_worker=0, seed=0))
