#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=8:00:00
source activate /work/frink/mcinerney.de/envs/ehragent
all_sentences="/scratch/mcinerney.de/ehr-diagnosis-agent-output/wandb/run-20230925_104538-57jsyt8v/files/ckpt_epoch=145_updates=1160.pt"
llm_evidence="/scratch/mcinerney.de/ehr-diagnosis-agent-output/wandb/run-20230927_194029-jhtf7taf/files/ckpt_epoch=95_updates=679.pt"
llm_evidence_bert_predictor="/scratch/mcinerney.de/ehr-diagnosis-agent-output/wandb/run-20231129_201913-pmgcy2p1/files/ckpt_epoch=150_updates=1200.pt"
longformer_predictor="/scratch/mcinerney.de/ehr-diagnosis-agent-output/wandb/run-20231202_180041-4hxu38wc/files/ckpt_epoch=85_updates=680.pt"
bert_predictor="/scratch/mcinerney.de/ehr-diagnosis-agent-output/wandb/run-20231203_090527-flsmcmkx/files/ckpt_epoch=65_updates=520.pt"
# python eval.py eval.split=test eval.checkpoint=$all_sentences actor.shared_params.use_raw_sentences=true eval.seed_offset=0
# python eval.py eval.split=test eval.checkpoint=$all_sentences actor.shared_params.use_raw_sentences=true eval.seed_offset=1
# python eval.py eval.split=test eval.checkpoint=$all_sentences actor.shared_params.use_raw_sentences=true eval.seed_offset=2
# python eval.py eval.split=test eval.checkpoint=$all_sentences actor.shared_params.use_raw_sentences=true eval.seed_offset=3
# python eval.py eval.split=test eval.checkpoint=$all_sentences actor.shared_params.use_raw_sentences=true eval.seed_offset=4
# python eval.py eval.split=test eval.checkpoint=$llm_evidence eval.seed_offset=0
# python eval.py eval.split=test eval.checkpoint=$llm_evidence eval.seed_offset=1
# python eval.py eval.split=test eval.checkpoint=$llm_evidence eval.seed_offset=2
# python eval.py eval.split=test eval.checkpoint=$llm_evidence eval.seed_offset=3
# python eval.py eval.split=test eval.checkpoint=$llm_evidence eval.seed_offset=4
# python eval.py eval.split=test eval.checkpoint=$llm_evidence_bert_predictor actor.shared_params.embedder_type=bert eval.seed_offset=0
# python eval.py eval.split=test eval.checkpoint=$llm_evidence_bert_predictor actor.shared_params.embedder_type=bert eval.seed_offset=1
# python eval.py eval.split=test eval.checkpoint=$llm_evidence_bert_predictor actor.shared_params.embedder_type=bert eval.seed_offset=2
# python eval.py eval.split=test eval.checkpoint=$llm_evidence_bert_predictor actor.shared_params.embedder_type=bert eval.seed_offset=3
# python eval.py eval.split=test eval.checkpoint=$llm_evidence_bert_predictor actor.shared_params.embedder_type=bert eval.seed_offset=4
# python eval.py eval.split=test eval.checkpoint=$bert_predictor actor.shared_params.embedder_type=bert actor.shared_params.use_raw_sentences=true eval.seed_offset=0
# python eval.py eval.split=test eval.checkpoint=$bert_predictor actor.shared_params.embedder_type=bert actor.shared_params.use_raw_sentences=true eval.seed_offset=1
# python eval.py eval.split=test eval.checkpoint=$bert_predictor actor.shared_params.embedder_type=bert actor.shared_params.use_raw_sentences=true eval.seed_offset=2
# python eval.py eval.split=test eval.checkpoint=$bert_predictor actor.shared_params.embedder_type=bert actor.shared_params.use_raw_sentences=true eval.seed_offset=3
# python eval.py eval.split=test eval.checkpoint=$bert_predictor actor.shared_params.embedder_type=bert actor.shared_params.use_raw_sentences=true eval.seed_offset=4
python eval.py eval.split=test eval.checkpoint=$longformer_predictor actor.shared_params.embedder_type=bert actor.shared_params.use_raw_sentences=true actor.shared_params.model_name=yikuan8/Clinical-Longformer eval.seed_offset=0
python eval.py eval.split=test eval.checkpoint=$longformer_predictor actor.shared_params.embedder_type=bert actor.shared_params.use_raw_sentences=true actor.shared_params.model_name=yikuan8/Clinical-Longformer eval.seed_offset=1
python eval.py eval.split=test eval.checkpoint=$longformer_predictor actor.shared_params.embedder_type=bert actor.shared_params.use_raw_sentences=true actor.shared_params.model_name=yikuan8/Clinical-Longformer eval.seed_offset=2
python eval.py eval.split=test eval.checkpoint=$longformer_predictor actor.shared_params.embedder_type=bert actor.shared_params.use_raw_sentences=true actor.shared_params.model_name=yikuan8/Clinical-Longformer eval.seed_offset=3
python eval.py eval.split=test eval.checkpoint=$longformer_predictor actor.shared_params.embedder_type=bert actor.shared_params.use_raw_sentences=true actor.shared_params.model_name=yikuan8/Clinical-Longformer eval.seed_offset=4
