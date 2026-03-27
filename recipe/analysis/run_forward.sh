cd $HOME/train-verl-updated
torchrun --standalone --nnodes=1 --nproc_per_node=8 -m recipe.analysis.fsdp_forward $FORWARD_CONFIG