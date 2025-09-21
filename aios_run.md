# AiOS 

In order to run the AiOS model, please follow these steps:

1. Clone the AiOS repository from [GitHub](https://github.com/SMPLCap/AiOS?tab=readme-ov-file):

   ```bash
   git clone https://github.com/SMPLCap/AiOS.git
    ```

    and place it as described in the `README` file of this repository.
2. Create a new environment and install the dependencies following the instructions on the official repo. On windows the installation of `pytorch3d` might be tricky, please refer to the [official instructions](https://pytorch3d.org/docs/installation) for more details or just drop it since the 3D visualization will be done using `aitviewer`.
Notes:
- Make sure to remove `pickle5` from `requirements.txt` if it causes issues (it is python built-in).
- Since we are only using it for inference, you can avoid using the gpu toolkit in cuda thus comment lines 47/48 in `AiOS/models/aios/ops/setup.py`

3. Download the pretrained weights from [here](https://huggingface.co/ttxskk/AiOS/tree/main) (AiOS model) and place them in a folder called `checkpoint` inside the `AiOS/data` folder.

4. Make sure to also have the SMPL-X models in the correct folder as described in the `README` file of this repository.
