import os
import sys
import pickle
import numpy as np
from PIL import Image
import argparse
import torch

def run_PTI(args):
    from configs import paths_config, hyperparameters, global_config
    from utils.align_data import pre_process_images
    from scripts.run_pti import run_PTI
    from scripts.latent_editor_wrapper import LatentEditorWrapper
    
    # config setup
    image_dir_name = args.image_dir
    image_name = args.image_name
    use_multi_id_training =False
    global_config.device='cuda'
    paths_config.e4e = f'./pretrained_models/e4e_ffhq_encode.pt'
    paths_config.input_data_id = image_dir_name
    paths_config.input_data_path = f'./{image_dir_name}_processed'
    paths_config.stylegan2_ada_ffhq = f'./pretrained_models/ffhq.pkl'
    paths_config.checkpoints_dir = f'./checkpoints/PTI'
    paths_config.style_clip_pretrained_mapppers = f'./pretrained_models'
    hyperparameters.use_localtity_regularization = False

    # processed data
    original_image_dir = f'./{image_dir_name}'
    processed_image_dir = f'./{image_dir_name}_processed'# same as paths_config.input_data_path
    print(f"create processed: {processed_image_dir}")
    os.makedirs(processed_image_dir,exist_ok=True)

    original_image = Image.open(f'{original_image_dir}/{image_name}')
    pre_process_images(f'{original_image_dir}')
    aligned_image =  Image.open(f"{processed_image_dir}/{image_name.split('.')[0]}.jpeg")
    # run PTI
    model_id = run_PTI(use_wandb=False,use_multi_id_training=use_multi_id_training)

    # save latent code and Model weights
    with open(paths_config.stylegan2_ada_ffhq,'rb') as f:
        old_G = pickle.load(f)['G_ema'].cuda()
    with open(f'{paths_config.checkpoints_dir}/model_{model_id}_{image_name.split(".")[0]}.pt','rb') as f_new:
        new_G = torch.load(f_new).cuda()

    # export update pickle file
    with open(paths_config.stylegan2_ada_ffhq,'rb') as f:
        d = pickle.load(f)
        old_G = d['G_ema'].cuda()#.eval().requires_grad_(False).cpu()
        old_D = d['D'].eval().requires_grad_(False).cpu()

    tmp = {}
    tmp['G'] = old_G.eval().requires_grad_(False).cpu()
    tmp['G_ema'] = new_G.eval().requires_grad_(False).cpu()
    tmp['D'] = old_D
    tmp['training_set_kwargs'] = None
    tmp['augment_pipe'] = None

    with open(f'{paths_config.checkpoints_dir}/model_{model_id}.pkl', 'wb') as f:
        pickle.dump(tmp,f)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir',default='image_original',help='image dir of images to get latent code')
    parser.add_argument('--image_name',default='test.jpg',help='image to edit')
    args = parser.parse_args()
    
    #prepare_environ()
    #import sys
    #sys.path.append(f'{os.getcwd()}/{CODE_DIR}/')
    run_PTI(args)