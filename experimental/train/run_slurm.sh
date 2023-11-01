
# Train general-domain model from scratch
# We first try learning rate 1e-3, then lower to 8e-4 if the loss diverges

bash train/run_pretrain_pipeline_general.sh \
      wiki_and_books \
      dsir_scratch \
      60200 \
      /path/to/data.jsonl \
     "true" "true" "--from_scratch --adam_beta1 0.9 --adam_beta2 0.98 --adam_eps 1e-6 --max_grad_norm 1.0" 8e-4
