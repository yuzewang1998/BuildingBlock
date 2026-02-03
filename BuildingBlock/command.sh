#----------------------------------------------------------------------------------------------------------------------------------------------------------
# generate indoor layout (need 3d_future models,refer to ATISS/Diffuscene)
CUDA_VISIBLE_DEVICES=0 xvfb-run --server-num=103 python  generate_diffusion_scene_batch.py  \
    ../config/uncond/diffusion_bedrooms_instancond_lat32_v_T.yaml \
    ../sample/bedrooms ../3d_front_processed/bedrooms/threed_future_model_bedroom.pkl\
    --weight_file ../buildingBlock_indoor.ckpt \
    --without_screen  --n_sequences 100  --render_top2down --save_mesh --no_texture --without_floor  --clip_denoised --retrive_objfeats
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# generate building layout
CUDA_VISIBLE_DEVICES=0 python generate_diffusion_building.py "../config/uncond/diffusion_building_DIT.yaml" \
 --weight_file ../buildingBlock_building.ckpt  --n_sequences 100 --clip_denoised \
 --save_path ../sample/buildings
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#train uncond indoor layout generator
CUDA_VISIBLE_DEVICES=0 xvfb-run --server-num=100 python train_diffusion_scene.py ../config/uncond/diffusion_bedrooms_instancond_lat32_v_T.yaml  \
    indoor_uncond ../3d_front_processed/bedrooms/threed_future_model_bedroom.pkl  \
    --experiment_tag indoor_uncond --n_processes 2 --without_screen  --n_sequences 10  --render_top2down --no_texture \
    --without_floor  --clip_denoised --retrive_objfeats --fix_order
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#train uncond building layout generator
python train_diffusion_building_DDP.py ../config/text/diffusion_building_DIT.yaml uncond  --experiment_tag uncond --n_processes 0 --with_swanlab_logger
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#train text to building layout
 python train_diffusion_building_DDP.py ../config/text/diffusion_building_DIT.yaml \
 ../runs/text/t2l --experiment_tag t2l --n_processes 0 --with_wandb_logger
#----------------------------------------------------------------------------------------------------------------------------------------------------------
