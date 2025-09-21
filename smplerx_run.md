# SMPLer-X

## Set-up
In order to launch and visualize results in SMPLer-X, few modifications have to be done to the code. 

As a first thing create a dedicated environment:
```bash
conda create -n smplerx_env python=3.8
conda activate smplerx_env
```

Start by cloning the [SMPLer-X repository](https://github.com/SMPLCap/SMPLer-X) and the [emdb repository](https://github.com/eth-ait/emdb) following the structure in the `README` and installing the required dependencies:

In order to run the full pipeline, you also need to add two paths to the `PYTHONPATH` variable:
```bash
conda env config vars set PYTHONPATH="$env:PYTHONPATH;path_to_your_root\SMPLer-X\main\transformer_utils;path_to_your_root\emdb"
conda deactivate
conda activate smplerx_env
```

Make sure that you have the [smplx](https://smpl-x.is.tue.mpg.de/) models downloaded in `path_to_your_root\data\smplx_models\smplx`. And also make sure to install a couple of additional models:
- your favourite version of [SMPLer-X](https://huggingface.co/camenduru/SMPLer-X/tree/main) in `path_to_your_root\SMPLer-X\pretrained_models`
- your favourite versione of the [vitpose model](https://huggingface.co/camenduru/SMPLer-X/tree/main)
- The `mmdet` models that you can find in the [SMPLer-X repository](https://github.com/SMPLCap/SMPLer-X?tab=readme-ov-file)

Additional modifications have to be done:
- Add an empty `__init__.py` file in the `emdb` folder
- change line 18 in `path_to_your_root\SMPLer-X\main\config.py` to:
    ```python        
    self.human_model_path = osp.join(self.root_dir,"..", "data","smplx_models")
    ```
- change the function update_config in `path_to_your_root\SMPLer-X\main\config.py` to:
    ```python
    def update_config(self, num_gpus, exp_name):
            self.num_gpus = num_gpus
            self.exp_name = exp_name
            
            self.prepare_dirs(self.exp_name)
            
            # Save
            cfg_save = MMConfig(self.__dict__)
            save_path = osp.join(self.code_dir, 'config_base.py')
            save_path = save_path.replace("\\", "/") 
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(repr(cfg_save._cfg_dict))
    ```
- In the config file related to your model (e.g. `path_to_your_root\SMPLer-X\main\config\config_smpler_x_s32.py`) change the line:
    ```python
    encoder_pretrained_model_path = '../pretrained_models/vitpose_small.pth'
    ```
    making sure to include your downloaded vitpose model (in my case `vitpose_small`).
- Since the visualization is done using `aitviewer`, make sure to install it using
    ```bash
    pip install aitviewer
    ```
    and you can also comment, in `path_to_your_root\SMPLer-X\inference.py`, the lines 164 and 165:
    ```python
    vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, 
                                  mesh_as_vertices=args.show_verts)
    ```

## Visualizing a video
In order to visualize results for a specific video, navigate to `EMDBX\scripts\smplerx.py` and modify the following variables according to the required format and the pretrained SMPLer-X model you want to use:
```python
video_name = "P0_00_mvs_a_video"
video_format = "mp4"
fps = 30
ckpt = "smpler_x_s32"
```
And also change the following:
```python
smplerx_folder = r"path_to_root\SMPLer-X\demo\results\P0_00_mvs_a_video\smplx"
output_path = r"path_to_root\EMDBX\smplerx_outputs"
sequence_name = "P0_00_mvs_a_video"
```
Moreover you need to have saved the video from which you want to extract the frames in `path_to_root\SMPLer-X\demo\videos`.

Finally, you can run the script by launching:
```bash
EMDBX\scripts > python smplerx.py
```