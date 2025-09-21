import os 
import subprocess
import glob
import pickle as pkl
import numpy as np
import shutil
import logging

logging.basicConfig(level=logging.INFO)

from aitviewer.configuration import CONFIG as C
from smplerx_visualizer import main as visualize_smplerx_main
from argparse import Namespace
from scipy.spatial.transform import Rotation as R

class SMPLEREXLauncher:
    def __init__(
        self, 
        sequence_name: str,
        person_name: str,
        video_format: str = "mp4",
        fps: int = 30,
        ckpt: str = "smpler_x_s32"
    ):
        self.sequence_name = sequence_name
        self.person_name = person_name
        self.video_format = video_format
        self.fps = fps
        self.ckpt = ckpt
        self.smplerx_folder = f"C:/Users/tommy/OneDrive/Documenti/Tommy/ETH/Semester project/code/SMPLer-X/demo/results/{self.sequence_name}/smplx"
        self.output_path = f"C:/Users/tommy/OneDrive/Documenti/Tommy/ETH/Semester project/code/EMDBX/smplerx_outputs/{self.person_name}_{self.sequence_name}"
        self.emdb_sequence_path = os.path.join("..", "..", "data", "EMDB_dataset", self.person_name, self.sequence_name)
        self.base_dir =os.path.join("..", "..", "SMPLer-X")


    def launch_smplerx(self):

        video_path = os.path.join(self.base_dir, os.path.join("demo", "videos"), f"{self.person_name}_{self.sequence_name}_video.{self.video_format}")
        img_path = os.path.join(self.base_dir, os.path.join("demo", "images"), self.sequence_name)
        save_dir = os.path.join(self.base_dir, os.path.join("demo", "results"), self.sequence_name)

        os.makedirs(img_path, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-f", "image2",
            "-vf", f"fps={self.fps}/1",
            "-qscale", "0",
            os.path.join(img_path, "%06d.jpg")
        ])

        n_frames = len(glob.glob(os.path.join(img_path, "*.jpg")))
        logging.info("Number frames:", n_frames)

        inference_script = os.path.join(self.base_dir, "main", "inference.py")

        subprocess.run([
            "python", inference_script,
            "--num_gpus", "1",
            "--exp_name", os.path.join("output", f"demo_{self.sequence_name}"),
            "--pretrained_model", self.ckpt,
            "--agora_benchmark", "agora_model",
            "--img_path", img_path,
            "--start", "1",
            "--end", str(n_frames),
            "--output_folder", save_dir,
            # "--show_verts",
            # "--show_bbox",
            "--save_mesh"
        ], check = True, cwd=os.path.join(self.base_dir, "main"))

        subprocess.run([
            "ffmpeg", "-y", "-f", "image2", "-r", str(self.fps),
            "-i", os.path.join(save_dir, "img", "%06d.jpg"),
            "-vcodec", "mjpeg", "-qscale", "0", "-pix_fmt", "yuv420p",
            os.path.join(self.base_dir, "demo/results", f"{self.sequence_name}.mp4")
        ])

    def postprocess_smplerx_to_viewer(self):
        """Convert SMPLer-X .npz outputs into EMDB-like pkl for visualization with aitviewer (SMPL-X)."""

        smplerx_files = sorted(glob.glob(os.path.join(self.smplerx_folder, "*.npz")))
        all_data = {}

        for f in smplerx_files:
            data = np.load(f)
            for key in data.files:
                arr = np.squeeze(data[key])
                if key not in all_data:
                    all_data[key] = []
                all_data[key].append(arr)

        # Merge into arrays of shape (F, D)
        merged = {k: np.stack(v, axis=0) for k, v in all_data.items()}
        n_frames = merged["body_pose"].shape[0]

        # Load camera parameters from the original EMDB sequence since SMPLER-X does not estimate them
        with open (os.path.join(self.emdb_sequence_path, f"{self.person_name}_{self.sequence_name}_data.pkl"), "rb") as f:
            emdb_data = pkl.load(f)
        merged["camera_intrinsics"] = emdb_data["camera"]["intrinsics"]
        merged["camera_extrinsics"] = emdb_data["camera"]["extrinsics"]
        merged["width"] = emdb_data["camera"]["width"]
        merged["height"] = emdb_data["camera"]["height"]

        # Adjust SMPL-X coordinate system to aitviewer
        flip_y = np.diag([1, -1, 1])
        merged["transl"] = (flip_y @ merged["transl"].T).T

        rotations = R.from_rotvec(merged["global_orient"])
        rots_corrected = R.from_matrix(flip_y)*rotations
        merged["global_orient"] = rots_corrected.as_rotvec()

        # Build EMDB-like dict but for SMPL-X
        emdb_like = {
            "gender": "neutral",
            "smplx": {
                "body_pose": merged["body_pose"].reshape(n_frames, -1),          # (F, 63)
                "global_orient": merged["global_orient"].reshape(n_frames, -1),  # (F, 3)
                "betas": merged["betas"].reshape(n_frames, -1),                  # (F, 10)
                "transl": merged["transl"].reshape(n_frames, -1),                # (F, 3)
                "left_hand_pose": merged.get("left_hand_pose", np.zeros((n_frames, 45))).reshape(n_frames, 45),   # (F, 45)
                "right_hand_pose": merged.get("right_hand_pose", np.zeros((n_frames, 45))).reshape(n_frames, 45),
                "jaw_pose": merged.get("jaw_pose", np.zeros((n_frames, 3))), # FIXME: unused so far
                "expression": merged.get("expression", np.zeros((n_frames, 10))), # FIXME: unused so far
            },
            "camera": {
                "intrinsics": merged["camera_intrinsics"],
                "extrinsics": merged["camera_extrinsics"],
                "width": merged["width"],
                "height": merged["height"],
            },
            "kp2d": None,
            "bboxes": {"bboxes": None},
        }

        # Save to pickle
        pkl_file = os.path.join(self.output_path, f"{self.person_name}_{self.sequence_name}_data.pkl")
        with open(pkl_file, "wb") as f:
            pkl.dump(emdb_like, f)

        logging.info(f"Saved {pkl_file} with {n_frames} frames.")

        src_img_dir = os.path.join(self.base_dir, "demo", "images", self.sequence_name)
        dst_img_dir = os.path.join(self.output_path, "images")
        os.makedirs(dst_img_dir, exist_ok=True)

        # for img_file in sorted(glob.glob(os.path.join(src_img_dir, "*.jpg"))):
        #     shutil.copy(img_file, dst_img_dir)

        # logging.info(f"Copied {len(glob.glob(os.path.join(dst_img_dir, '*.jpg')))} images to {dst_img_dir}")

        # Prepare args for visualizer
        args = Namespace(
            subject="P0",
            sequence="00",
            view_from_camera=True,
            draw_2d=False,
            draw_trajectories=False,
        )

        # Update SMPL-X model path
        C.update_conf({"smplx_models": "../../data/smplx_models"})

        # Run visualization
        visualize_smplerx_main(args, outputs_path=self.output_path, original_images_path=self.emdb_sequence_path)


if __name__ == "__main__":
    sequence_name = "00_mvs_a"
    person_name = "P0"
    smplerx = SMPLEREXLauncher(sequence_name=sequence_name, person_name=person_name)
    smplerx.launch_smplerx()
    smplerx.postprocess_smplerx_to_viewer()
