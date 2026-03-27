set -e
echo "${HOME}"
cd $HOME/train-verl-updated
export WANDB_DIR=$HOME/wandb_logs/
mkdir -p $WANDB_DIR
export WANDB_API_KEY=d0a97fe1ba84f9958aee7b38fef9ac05048af4e1


echo "$EXP_NAME"
python3 -m recipe.prefix_rft_v2.main $TRAIN_CONFIG