import os 
os.environ["PYOPENGL_PLATFORM"] = "pyglet"
import subprocess
import glob
import sys

def launch_smplerx():
    video_name = "P1_13_outdoor_long_walk_video"
    video_format = "mp4"
    fps = 30
    ckpt = "smpler_x_h32"

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


if __name__ == "__main__":
    launch_smplerx()