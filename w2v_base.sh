#! /usr/local/bin/sbatch

#SBATCH --job-name=w2v_base
#SBATCH --output=/checkpoint/anuroops/fairseq/wav2vec/tmp/w2v_base-%A_%a.out
#SBATCH --error=/checkpoint/anuroops/fairseq/wav2vec/tmp/w2v_base-%A_%a.err
#SBATCH --partition=learnfair
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
#SBATCH --open-mode=append
#SBATCH --time=1000
#SBATCH --mem=500000
#SBATCH --constraint=volta32gb



trap_handler () {
   echo "Caught signal: " $1
   # SIGTERM must be bypassed
   if [ "$1" = "TERM" ]; then
       echo "bypass sigterm"
   else
     # Submit a new job to the queue
     echo "Requeuing " $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
     # SLURM_JOB_ID is a unique representation of the job, equivalent
     # to above
     scontrol requeue $SLURM_JOB_ID
   fi
}


# Install signal handler
trap 'trap_handler USR1' USR1
trap 'trap_handler TERM' TERM

srun python train.py /checkpoint/anuroops/data/libris/proc/ \
  --save-dir /checkpoint/anuroops/fairseq/wav2vec/tmp --fp16 \
  --num-workers 6 --task audio_pretraining --criterion wav2vec --arch wav2vec2 \
  --log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets \
  --extractor-mode default \
  --conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' \
  --final-dim 256 --latent-vars 320 --latent-groups 2 --latent-temp '(2,0.5,0.999995)' \
  --infonce --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay \
  --total-num-update 400000 --lr 0.0005 --warmup-updates 32000 --mask-length 10 --mask-prob 0.65 \
  --mask-selection static --mask-other 0 --encoder-layerdrop 0.05 --dropout-input 0.1 \
  --dropout-features 0.1 --feature-grad-mult 0.1 --loss-weights '[0.1, 10]' --conv-pos 128 \
  --conv-pos-groups 16 --num-negatives 100 --cross-sample-negatives 0 --max-sample-size 250000 \
  --min-sample-size 32000 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
  --max-tokens 1400000 --max-update 400000 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d

wait $!
