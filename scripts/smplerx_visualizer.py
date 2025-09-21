# visualize_smplx.py
"""
Visualizer adapted for SMPL-X sequences.
Maintains same structure as EMDB visualize.py but supports SMPL-X parameters.
"""

import argparse
import glob
import os
import pickle as pkl

import cv2
import numpy as np
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.lines import LinesTrail
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.viewer import Viewer

from emdb.configuration import (
    EMDB_ROOT,
    SMPL_SIDE_COLOR,
    SMPL_SIDE_INDEX,
    SMPL_SKELETON,
    SMPLX_MODELS,
)


def draw_kp2d(kp2d, bboxes=None):
    def _draw_kp2d(img, current_frame_id):
        current_kp2d = kp2d[current_frame_id].copy()
        scale = img.shape[0] / 1000

        for index in range(SMPL_SKELETON.shape[0]):
            i, j = SMPL_SKELETON[index]
            cv2.line(
                img,
                tuple(current_kp2d[i, :2].astype(np.int32)),
                tuple(current_kp2d[j, :2].astype(np.int32)),
                (0, 0, 0),
                int(scale * 3),
            )

        for jth in range(0, kp2d.shape[1]):
            color = SMPL_SIDE_COLOR[SMPL_SIDE_INDEX[jth]]
            radius = scale * 5
            img = cv2.circle(
                img,
                tuple(current_kp2d[jth, :2].astype(np.int32)),
                int(radius * 1.4),
                (0, 0, 0),
                -1,
            )
            img = cv2.circle(
                img,
                tuple(current_kp2d[jth, :2].astype(np.int32)),
                int(radius),
                color,
                -1,
            )

        if bboxes is not None:
            bbox = bboxes[current_frame_id]
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        return img

    return _draw_kp2d


def draw_nothing(kp2d, bboxes=None):
    def _draw_nothing(img, current_frame_id):
        return img

    return _draw_nothing


def get_camera_position(Rt):
    pos = -np.transpose(Rt[:, :3, :3], axes=(0, 2, 1)) @ Rt[:, :3, 3:]
    return pos.squeeze(-1)


def get_sequence_root(args):
    sequence_id = "{:0>2d}".format(int(args.sequence))
    candidates = glob.glob(os.path.join(EMDB_ROOT, args.subject, sequence_id + "*"))
    if len(candidates) == 0:
        raise ValueError(f"Could not find sequence {args.sequence} for subject {args.subject}.")
    elif len(candidates) > 1:
        raise ValueError(f"Sequence ID {args.sequence}* for subject {args.subject} is ambiguous.")
    return candidates[0]


def main(args, outputs_path, original_images_path):
    sequence_root = outputs_path
    data_file = glob.glob(os.path.join(sequence_root, "*_data.pkl"))[0]
    with open(data_file, "rb") as f:
        data = pkl.load(f)

    # SMPL-X layer
    gender = data.get("gender", "neutral")
    smplx_layer = SMPLLayer(model_type="smplx", gender=gender)

    # Create SMPL-X sequence
    smpl_seq = SMPLSequence(
        poses_body=data["smplx"]["body_pose"],
        smpl_layer=smplx_layer,
        poses_root=data["smplx"]["global_orient"],
        betas=data["smplx"]["betas"].reshape((1, -1)),
        trans=data["smplx"]["transl"],
        poses_left_hand=data["smplx"]["left_hand_pose"],
        poses_right_hand=data["smplx"]["right_hand_pose"],
        # poses_jaw=data["smplx"]["jaw_pose"],
        # expression=data["smplx"]["expression"],
        name="SMPL-X Fit",
    )

    kp2d = data["kp2d"]
    bboxes = data["bboxes"]["bboxes"]
    drawing_function = draw_kp2d if args.draw_2d else draw_nothing

    image_dir = os.path.join(original_images_path, "images")
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    intrinsics = data["camera"]["intrinsics"]
    extrinsics = data["camera"]["extrinsics"]
    cols, rows = data["camera"]["width"], data["camera"]["height"]

    viewer_size = None
    if args.view_from_camera:
        target_height = 1080
        width = int(target_height * cols / rows)
        viewer_size = (width, target_height)
        args.draw_trajectories = False

    viewer = Viewer(size=viewer_size)

    intrinsics = np.repeat(intrinsics[np.newaxis, :, :], len(extrinsics), axis=0)
    cameras = OpenCVCamera(intrinsics, extrinsics[:, :3], cols, rows, viewer=viewer, name="Camera")

    raw_images_bb = Billboard.from_camera_and_distance(
        cameras,
        10.0,
        cols,
        rows,
        image_files,
        image_process_fn=drawing_function(kp2d, bboxes),
        name="Image",
    )

    viewer.scene.add(smpl_seq, cameras, raw_images_bb)

    if args.draw_trajectories:
        smpl_path = LinesTrail(
            smpl_seq.joints[:, 0],
            r_base=0.003,
            color=(0, 0, 1, 1),
            cast_shadow=False,
            name="SMPL-X Trajectory",
        )

        cam_pos = get_camera_position(extrinsics)
        camera_path = LinesTrail(
            cam_pos,
            r_base=0.003,
            color=(0.5, 0.5, 0.5, 1),
            cast_shadow=False,
            name="Camera Trajectory",
        )

        viewer.scene.add(smpl_path, camera_path)

    if args.view_from_camera:
        viewer.set_temp_camera(cameras)
        viewer.render_gui = False
    else:
        viewer.center_view_on_node(smpl_seq)

    viewer.scene.origin.enabled = False
    viewer.scene.floor.enabled = False
    viewer.playback_fps = 30.0

    viewer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", type=str, help="The subject ID, e.g. P0 - P9.")
    parser.add_argument("sequence", type=str, help="Sequence ID (prefix or full name).")
    parser.add_argument("--view_from_camera", action="store_true")
    parser.add_argument("--draw_2d", action="store_true")
    parser.add_argument("--draw_trajectories", action="store_true")
    args = parser.parse_args()

    C.update_conf({"smplx_models": SMPLX_MODELS})
    main(args)
