To use a pre trained model you would need to set the filepath to the following when training

CUDA_VISIBLE_DEVICES='0' python3 -u train_models_forward.py +args=config_Adv.yaml ++args.model_name='Unet' ++args.filename='../data/1D/Advection/Train/1D_Advection_Sols_beta4.0.hdf5' ++args.ar_mode=True ++args.pushforward=True ++args.unroll_step=20 ++args.if_training=False ++args.continue_training=True


So you would need to place 1D_Advection_Sols_beta4.0_Unet-PF-20.pt inside the folder ../data/1D/Advection/Train/ before running the above command.