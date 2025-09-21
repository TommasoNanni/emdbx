import os 
import subprocess
import glob
import pickle as pkl
import numpy as np

from aitviewer.configuration import CONFIG as C
from emdb.visualize import main as visualize_main
from argparse import Namespace

def launch_smplerx():
    video_name = "P0_00_mvs_a_video"
    video_format = "mp4"
    fps = 30
    ckpt = "smpler_x_s32"

    base_dir =os.path.join("..", "..", "SMPLer-X")
    video_path = os.path.join(base_dir, os.path.join("demo", "videos"), f"{video_name}.{video_format}")
    img_path = os.path.join(base_dir, os.path.join("demo", "images"), video_name)
    save_dir = os.path.join(base_dir, os.path.join("demo", "results"), video_name)

    os.makedirs(img_path, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-f", "image2",
        "-vf", f"fps={fps}/1",
        "-qscale", "0",
        os.path.join(img_path, "%06d.jpg")
    ])

    n_frames = len(glob.glob(os.path.join(img_path, "*.jpg")))
    print("Number frames:", n_frames)

    inference_script = os.path.join(base_dir, "main", "inference.py")

    subprocess.run([
        "python", inference_script,
        "--num_gpus", "1",
        "--exp_name", os.path.join("output", f"demo_{video_name}"),
        "--pretrained_model", ckpt,
        "--agora_benchmark", "agora_model",
        "--img_path", img_path,
        "--start", "1",
        "--end", str(n_frames),
        "--output_folder", save_dir,
        # "--show_verts",
        # "--show_bbox",
        "--save_mesh"
    ], check = True, cwd=os.path.join(base_dir, "main"))

    subprocess.run([
        "ffmpeg", "-y", "-f", "image2", "-r", str(fps),
        "-i", os.path.join(save_dir, "img", "%06d.jpg"),
        "-vcodec", "mjpeg", "-qscale", "0", "-pix_fmt", "yuv420p",
        os.path.join(base_dir, "demo/results", f"{video_name}.mp4")
    ])


def postprocess_smplerx_to_viewer(smplerx_folder: str, output_path: str, sequence_name: str):
    smplerx_files = sorted(glob.glob(os.path.join(smplerx_folder, "*.npz")))
    all_data = {}

    for f in smplerx_files:
        data = np.load(f)
        for key in data.files:
            arr = np.squeeze(data[key])
            if key not in all_data:
                all_data[key] = []
            all_data[key].append(arr)

    print(all_data.keys())
    merged = {k: np.stack(v, axis=0) for k, v in all_data.items()}
    n_frames = merged["body_pose"].shape[0]

    emdb_like = {
        "gender": "neutral",
        "smpl": {
            "poses_body": merged["body_pose"].reshape(n_frames, -1),   # (F, 63)
            "poses_root": merged["global_orient"].reshape(n_frames, -1),  # (F, 3)
            "betas": merged["betas"].reshape(n_frames, -1),  # (F, 10)
            "trans": merged["transl"].reshape(n_frames, -1),  # (F, 3)
        },
        "camera": {
            "intrinsics": np.eye(3),
            "extrinsics": np.repeat(np.eye(4)[None, ...], n_frames, axis=0),
            "width": 640,
            "height": 480,
        },
        "kp2d": None,
        "bboxes": {"bboxes": None},
    }

    out_file = os.path.join(output_path, f"{sequence_name}_data.pkl")
    with open(out_file, "wb") as f:
        pkl.dump(emdb_like, f)

    print(f"Saved {out_file} with {n_frames} frames.")

    args = Namespace(
        subject = "P0",
        sequence = "00",
        view_from_camera = False,
        draw_2d = False,
        draw_trajectories = False,
    )
    C.update_conf({"smplx_models": "../../data/smplx_models"})

    visualize_main(args)


if __name__ == "__main__":
    smplerx_folder = r"C:\Users\tommy\OneDrive\Documenti\Tommy\ETH\Semester project\code\SMPLer-X\demo\results\P0_00_mvs_a_video\smplx"
    output_path = r"C:\Users\tommy\OneDrive\Documenti\Tommy\ETH\Semester project\code\EMDBX\smplerx_outputs"
    sequence_name = "P0_00_mvs_a_video"
    launch_smplerx()
    postprocess_smplerx_to_viewer(smplerx_folder, output_path, sequence_name)
    
    