import subprocess

class AiosLauncher:
    def __init__(
        self,
        config: str = "config/aios_smplx_demo.py",
        batch_size: int = 8,
        backbone: str = "resnet50"
    ):
        """
        Initialize the AIOS inference launcher.

        Args:
            config (str): Path to the config file.
            batch_size (int): Batch size for inference.
            backbone (str): Backbone architecture (e.g., resnet50).
        """
        self.config = config
        self.batch_size = batch_size
        self.backbone = backbone

    def run_inference(
        self,
        checkpoint: str,
        input_video: str,
        output_dir: str,
        num_person: int = 1,
        threshold: float = 0.3,
        gpu_num: int = 8
    ):
        """
        Run AIOS inference using torch.distributed.launch.

        Args:
            checkpoint (str): Path to the checkpoint (.pth).
            input_video (str): Path to the input video.
            output_dir (str): Output directory (will be created under demo/).
            num_person (int): Max number of people to estimate.
            threshold (float): Detection threshold.
            gpu_num (int): Number of GPUs to use with torch.distributed.
        """
        cmd = [
            "python", "-m", "torch.distributed.launch",
            f"--nproc_per_node={gpu_num}",
            "main.py",
            "-c", self.config,
            "--options",
            f"batch_size={self.batch_size}",
            f"backbone={self.backbone}",
            f"num_person={num_person}",
            f"threshold={threshold}",
            "--resume", checkpoint,
            "--eval",
            "--inference",
            "--to_vid",
            "--inference_input", input_video,
            "--output_dir", f"demo/{output_dir}"
        ]

        print("Running command:", " ".join(cmd))
        subprocess.run(cmd, check=True)
