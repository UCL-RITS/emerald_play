#BSUB -o /home/ucl/%(username)s/%(remote_results_path)s/%(project)s/%%J.log
#BSUB -e /home/ucl/%(username)s/%(remote_results_path)s/%(project)s/%%J.err
#BSUB -W 00:05
#BSUB -n 4
mpirun -lsf /home/ucl/%(username)s/%(remote_install_path)s/%(project)s/bin/%(project)s