set -e
echo "${HOME}"
cd $HOME/train-verl-updated
export WANDB_DIR=$HOME/wandb_logs/
mkdir -p $WANDB_DIR
export WANDB_API_KEY=YOUR_WANDB_API_KEY


echo "$EXP_NAME"
python3 -m recipe.prefix_rft.main $TRAIN_CONFIG