from aitviewer.configuration import CONFIG as C
from aitviewer.viewer import Viewer
from aitviewer.renderables.smpl import SMPLSequence, SMPLLayer
from aitviewer.remote.viewer import RemoteViewer
from aitviewer.remote.renderables.meshes import RemoteMeshes
import numpy as np
import os
import glob
import cv2

from aitviewer.renderables.billboard import Billboard
from aitviewer.scene.camera import OpenCVCamera

def t_pose():
    smpl_layer = SMPLLayer(model_type="smplx", gender="neutral")
    poses = np.zeros([1, smpl_layer.bm.NUM_BODY_JOINTS * 3])
    smpl_seq = SMPLSequence(poses, smpl_layer)

    v = Viewer()
    v.scene.add(smpl_seq)
    v.run()

def data_from_disk():
    sequence_name = "P1_14_outdoor_climb"
    load_path = r"C:\Users\tommy\OneDrive\Documenti\Tommy\ETH\Semester project\code\data\EMDB_dataset\_P1\P1\14_outdoor_climb"
    pkl_path = os.path.join(load_path, f"{sequence_name}_data.pkl")
    smpl_layer = SMPLLayer(model_type="smplx", gender="neutral")

    poses = np.zeros([1, smpl_layer.bm.NUM_BODY_JOINTS * 3])
    seqs_3dpw, camera_info = SMPLSequence(poses, smpl_layer).from_3dpw(
        pkl_data_path=pkl_path, name=sequence_name, color=(24 / 255, 106 / 255, 153 / 255, 1.0)
    )

    images_path = os.path.join(load_path, "\images")
    images = sorted(glob.glob(os.path.join(images_path, "*.jpg")))
    image0 = cv2.imread(images[0])
    cols, rows = image0.shape[1], image0.shape[0]

    v = Viewer(size=(cols // 2, rows // 2))
    v.playback_fps = 30.0
    v.scene.floor.enabled = False
    v.scene.origin.enabled = False

    cam = OpenCVCamera(camera_info["intrinsics"], camera_info["extrinsics"][:, :3], cols=cols, rows=rows, viewer=v)
    billboard = Billboard.from_camera_and_distance(cam, 15.0, cols=cols, rows=rows, textures=images)
    v.scene.add(*seqs_3dpw, cam, billboard)
    
    v.set_temp_camera(cam)
    v.run()

if __name__ == "__main__":
    data_from_disk()