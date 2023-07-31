#!/bin/bash

# Start the first process
bash vncorenlp_service.sh &
  
# Start the second process
PYTHONPATH=./ PATH=/root/miniconda3/envs/nmt/bin:/root/miniconda3/bin:/root/miniconda3/condabin:/root/miniconda3/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin python app.py &
  
# Wait for any process to exit
wait -n
  
# Exit with status of process that exited first
exit $?