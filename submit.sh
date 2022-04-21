
sbatch \
  --job-name H_UROP \
  --partition=gpu \
  --gres=gpu:1 \
  --output std_out.log \
  --array 1-3 \
  $(pwd -P)/example_run2.sh

  #asks for one gpu, you can set this to max 8 if you need more.
  #this is a text output file. It will be overwritten each time you run this command.
  
  #--array 1-10 \ #use this if you need more than one job instance.
  # # It will set the variable
  # # ${SLURM_ARRAY_TASK_ID}
  # # in your bash script


