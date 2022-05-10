from pytools.persistent_dict import PersistentDict
from time import sleep
from phasegrok.definitions import Ranges, Locations

gpu_jobs = PersistentDict("jobs")

onstart:
  gpu_jobs.store("gpu0", 0)
  gpu_jobs.store("gpu1", 0)

def run_on_free_gpu(cmd):
  while True:
    job_counts = [gpu_jobs.fetch("gpu0"), gpu_jobs.fetch("gpu1")]
    if job_counts[0] >= 3 and job_counts[1] >= 3: # at most 3 jobs per gpu
      sleep(15)
      continue
    cuda_id = 0 if job_counts[0] <= job_counts[1] else 1
    gpu_jobs.store(f"gpu{cuda_id}", job_counts[cuda_id] + 1)
    print(f"running on GPU {cuda_id}")
    shell(cmd + f" --device cuda:{cuda_id}")
    gpu_jobs.store(f"gpu{cuda_id}", gpu_jobs.fetch(f"gpu{cuda_id}") - 1)
    break

def train_cmd(
  weight_decay=None,
  seed=None,
  dropout=None,
  decoder_lr=None,
  esam_rho=None,
):
    return " ".join(
        [
            f"python scripts/transformer_grokking.py",
            f"--weight_decay {weight_decay}" if weight_decay is not None else "",
            f"--lr_decoder {decoder_lr}" if decoder_lr is not None else "",
            f"--seed {seed}" if seed is not None else "",
            f"--dropout {dropout}" if dropout is not None else "",
            f"--esam_rho {esam_rho}" if esam_rho is not None else "",
            f"--log 10",
            f"--epochs 100000",
            f"--stop_early",
        ]
    )

rule all:
  input:
    expand(Locations.weight_decay, wd=Ranges.weight_decay, seed=Ranges.seed),
    expand(Locations.dropout, do=Ranges.dropout, seed=Ranges.seed),
    expand(Locations.slow_decoder, lr=Ranges.decoder_lr, seed=Ranges.seed),
    # expand(Locations.esam, rho=esam_rhos, beta=[.5, 1], seed=Ranges.seed),
    # expand(Locations.weight_decay_vs_decoder_lr, wd=Ranges.weight_decay, lr=Ranges.decoder_lr, seed=[1]),


rule with_weight_decay:
  output: 
    logs = directory(Locations.weight_decay)
  run: 
    cmd = train_cmd(weight_decay=wildcards.wd, seed=wildcards.seed)
    run_on_free_gpu(cmd + f" --exp_name {output.logs}")

rule with_dropout:
  output: 
    logs = directory(Locations.dropout)
  run: 
    cmd = train_cmd(dropout=wildcards.do, seed=wildcards.seed)
    run_on_free_gpu(cmd + f" --exp_name {output.logs}")

rule with_slow_decoder:
  output: 
    logs = directory(Locations.slow_decoder)
  run: 
    cmd = train_cmd(decoder_lr=wildcards.lr, seed=wildcards.seed)
    run_on_free_gpu(cmd + f" --exp_name {output.logs}")

rule with_esam:
  output:
    logs = directory(Locations.esam)
  run:
    cmd = train_cmd(esam_rho=wildcards.rho, esam_beta=wildcards.beta, seed=wildcards.seed)
    run_on_free_gpu(cmd + f" --esam --exp_name {output.logs}")

rule weight_decay_vs_decoder_lr:
  output:
    logs = directory(Locations.weight_decay_vs_decoder_lr)
  run:
    cmd = train_cmd(weight_decay=wildcards.wd, decoder_lr=wildcards.lr, seed=wildcards.seed)
    run_on_free_gpu(cmd + f" --exp_name {output.logs}")
