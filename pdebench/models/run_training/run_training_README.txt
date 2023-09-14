When you run something like the following line:

CUDA_VISIBLE_DEVICES='0' python3 -u train_models_forward.py +args=config_diff-sorp.yaml ++args.ar_mode=False ++args.pushforward=False ++args.model_name='Unet' ++args.if_training=True


The following happens:

+args=config_diff-sorp.yaml    this overrides the default config file called config.yaml inside pdebench/models/config

So now config_diff-sorp.yaml contains all the configurations that will be run by train_models_forward.py

++args can now be used to override any arguments inside of the file config_diff-sorp.yaml