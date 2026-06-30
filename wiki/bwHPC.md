### File transfer
```sh
scp -c aes128-gcm@openssh.com <local> bw:<remote>
scp -c aes128-gcm@openssh.com bw:/pfs/work7/workspace/scratch/fr_ja162-data/data/PulseDB/Supplementary_Subset_Files/VitalDB_AAMI_Test_Subset.mat "D:\University\Semester\Bachelor Arbeit"
```

### SSH

Login Into Node directly from ssh
Maybe use --jobname=
```sh

ssh fr_ja162@uc2.scc.kit.edu srun --jobid=$(squeue --state='RUNNING' | awk 'NR==2{print $1}') --pty /usr/bin/bash
```

### TMUX
```sh
tmux list-sessions # list currently running sessions
tmux attach -t mySession12345 # open running tmux session 
```

### SLURM

Job state
```sh
squeue # shows running jobs
squeue --start # shows waiting time until ressource allocation
scontrol show jobid -dd <jobid> # get job information
```
Cancel Job
```sh
scancel -n "<job-name>""
scancel -t "<state>" # "PENDING", "RUNNING" or "SUSPENDED"
```
Node login, and second node:
```
srun --jobid=XXXXXXXX --pty /usr/bin/bash 
srun --nodelist=uc2nXXX --pty /bin/bash
```

Job starttime planning:
> The only reliable way would be to submit a job. Then Slurm can warn you by email with --mail-type=BEGIN.
> 
> Note that sbatch has a --test-only argument that tells you when your job would run if submitted, without actually submitting a job.
> 
> Also, srun has an --immediate argument that allows submitting and job and cancelling it if it does not get an allocation within a few seconds. sbatch has a similar parameter --deadline
> 
> Finally, if you need an interactive session and be available when the job starts, you can submit a job with --begin. For instance if you want to have an interactive session at the same time the next day, submit a job the day before (assuming reasonable job length) with --begin=now+24hours

### Enroot
Github login credentials
```
# ~/.config/enroot/.credentials

# Github Registry
machine ghcr.io login <user> password <token>
```
### Workspace
Reserve workspace named "data" for 60 days
```sh
ws_allocate -r 15 -m mail.allgeier@gmail.com data 60

ws_list # info about your workspaces
```
Find location of workspace "data"
```sh
ws_find data
```

```sh
ws_extend data 99 # extend workspace lifetime (by min(max_extension, 99)=60 days)

ws_release data # Manually erase your workspace
```

### Get real-time GPU usage information

```sh
watch -c -n 0.5 nvidia-smi
```
or
```sh
pip install gpustat
watch -c -n 0.5 gpustat -cp --color
```

Print current gpu stats (python)
```python
    print(torch.cuda.memory_summary())
```


### Direct development on the HPC nodes
Computing nodes should be only used for running code i.e. via jupyter. For simple file changes / development use the login server.
- https://docs.rc.fas.harvard.edu/kb/vscode-remote-development-via-ssh-or-tunnel/#Approach_II_Remote_8211_Tunnel_interactive
