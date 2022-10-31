#!/bin/bash
# Usage:
# sh roscoe_data/generate_perturbed_data.sh

PATH_TO_DATA="./projects/roscoe/roscoe_data/"
RESTORE_SCRIPT="./projects/roscoe/roscoe_data/restore_data.py"

mkdir -p ${PATH_TO_DATA}/synthetic
mkdir -p ${PATH_TO_DATA}/synthetic_sentinel

echo "Generating perturbations ..."
# ==== data generation for entailment_bank dataset
mkdir -p ${PATH_TO_DATA}/synthetic/entailment_bank_synthetic

for g in DuplicateOneStep ExtrinsicHallucinatedStep GrammaticalErrorStep NegateStep RemoveOneStep ShuffleSteps SwapOneStep
do
    python -m parlai.scripts.convert_data_to_json_format -t entailment_bank --step-by-step-style none --step-perturbations $g --extrinsic-step true -dt test --skip-perturbation-failures true
    mv tmp.jsonl ${PATH_TO_DATA}/synthetic/entailment_bank_synthetic/${g}_test.jsonl
done

# ==== data generation for asdiv dataset
mkdir -p ${PATH_TO_DATA}/synthetic/asdiv_synthetic

for g in ShuffleNumbers ShuffleOperations RandomNumber RandomOperation
do
    python -m parlai.scripts.convert_data_to_json_format -t asdiv --step-by-step-style none --math-step-perturbations $g -dt test --skip-perturbation-failures true
    mv tmp.jsonl ${PATH_TO_DATA}/synthetic/asdiv_synthetic/${g}_test.jsonl
done

# ==== data generation for math dataset
mkdir -p ${PATH_TO_DATA}/synthetic/math_dataset_synthetic

for g in DuplicateOneStep ExtrinsicHallucinatedStep GrammaticalErrorStep NegateStep RemoveOneStep ShuffleSteps SwapOneStep
do
    python -m parlai.scripts.convert_data_to_json_format -t math_dataset --step-by-step-style none --step-perturbations $g --extrinsic-step true -dt test --skip-perturbation-failures true
    mv tmp.jsonl ${PATH_TO_DATA}/synthetic/math_dataset_synthetic/${g}_test.jsonl
done
for g in RandomNumber RandomOperation ShuffleNumbers ShuffleOperations
do
    python -m parlai.scripts.convert_data_to_json_format -t math_dataset --step-by-step-style none --math-step-perturbations $g --extrinsic-step true -dt test --skip-perturbation-failures true
    mv tmp.jsonl ${PATH_TO_DATA}/synthetic/math_dataset_synthetic/${g}_test.jsonl
done

# ==== data generation for aqua dataset
mkdir -p ${PATH_TO_DATA}/synthetic/aqua_synthetic

for g in DuplicateOneStep ExtrinsicHallucinatedStep GrammaticalErrorStep NegateStep RemoveOneStep ShuffleSteps SwapOneStep
do
    python -m parlai.scripts.convert_data_to_json_format -t aqua:AQuAStepByStepReasoningTeacher --step-by-step-style none --step-perturbations $g --extrinsic-step true -dt test --skip-perturbation-failures true
    mv tmp.jsonl ${PATH_TO_DATA}/synthetic/aqua_synthetic/${g}_test.jsonl
done
for g in RandomNumber RandomOperation ShuffleNumbers ShuffleOperations
do
    python -m parlai.scripts.convert_data_to_json_format -t aqua:AQuAStepByStepReasoningTeacher --step-by-step-style none --math-step-perturbations $g --extrinsic-step true -dt test --skip-perturbation-failures true
    mv tmp.jsonl ${PATH_TO_DATA}/synthetic/aqua_synthetic/${g}_test.jsonl
done

# ==== data generation for eqasc dataset
mkdir -p ${PATH_TO_DATA}/synthetic/eqasc_synthetic

for g in DuplicateOneStep ExtrinsicHallucinatedStep GrammaticalErrorStep NegateStep RemoveOneStep
do
    python -m parlai.scripts.convert_data_to_json_format -t eqasc --step-by-step-style none --step-perturbations $g --extrinsic-step true -dt test --skip-perturbation-failures true
    mv tmp.jsonl ${PATH_TO_DATA}/synthetic/eqasc_synthetic/${g}_test.jsonl
done

# ==== data generation for proof_writer dataset
mkdir -p ${PATH_TO_DATA}/synthetic/proofwriter_synthetic

for g in DuplicateOneStep ExtrinsicHallucinatedStep GrammaticalErrorStep NegateStep RemoveOneStep ShuffleSteps SwapOneStep
do
    python -m parlai.scripts.convert_data_to_json_format -t proof_writer:ProofWriterStepByStepReasoningTeacher --step-by-step-style none --step-perturbations $g --extrinsic-step true --proofwriter-dataset depth-5 -dt test --skip-perturbation-failures true
    mv tmp.jsonl ${PATH_TO_DATA}/synthetic/proofwriter_synthetic/5${g}_test.jsonl
done

echo "Generating sentinel perturbations ..."
# ==== data generation for entailment_bank dataset sentenial errors
mkdir -p ${PATH_TO_DATA}/synthetic_sentinel/entailment_bank_synthetic

for g in DuplicateOneStep ExtrinsicHallucinatedStep GrammaticalErrorStep NegateStep ShuffleSteps SwapOneStep
do
    python -m parlai.scripts.convert_data_to_json_format -t entailment_bank --step-by-step-style none --step-perturbations DuplicateOneStep $g --extrinsic-step true -dt test --skip-perturbation-failures true
    mv tmp.jsonl ${PATH_TO_DATA}/synthetic_sentinel/entailment_bank_synthetic/DuplicateOneStep_${g}_test.jsonl
done

python -m parlai.scripts.convert_data_to_json_format -t entailment_bank --step-by-step-style none --step-perturbations SwapOneStep DuplicateOneStep ExtrinsicHallucinatedStep --extrinsic-step true -dt test --skip-perturbation-failures true
mv tmp.jsonl ${PATH_TO_DATA}/synthetic_sentinel/entailment_bank_synthetic/DuplicateOneStep_SwapOneStep_ExtrinsicHallucinatedStep_test.jsonl

# ==== data generation for math dataset sentenial errors
mkdir -p ${PATH_TO_DATA}/synthetic_sentinel/math_dataset_synthetic

for g in ExtrinsicHallucinatedStep GrammaticalErrorStep NegateStep
do
    python -m parlai.scripts.convert_data_to_json_format -t math_dataset --step-by-step-style none --step-perturbations DuplicateOneStep $g --extrinsic-step true -dt test --skip-perturbation-failures true
    mv tmp.jsonl ${PATH_TO_DATA}/synthetic_sentinel/math_dataset_synthetic/DuplicateOneStep_${g}_test.jsonl
done
for g in RandomNumber RandomOperation ShuffleNumbers ShuffleOperations
do
    python -m parlai.scripts.convert_data_to_json_format -t math_dataset --step-by-step-style none --step-perturbations DuplicateOneStep --math-step-perturbations $g --extrinsic-step true -dt test --skip-perturbation-failures true
    mv tmp.jsonl ${PATH_TO_DATA}/synthetic_sentinel/math_dataset_synthetic/DuplicateOneStep_${g}_test.jsonl
done

python -m parlai.scripts.convert_data_to_json_format -t math_dataset --step-by-step-style none --step-perturbations SwapOneStep DuplicateOneStep ExtrinsicHallucinatedStep --extrinsic-step true -dt test --skip-perturbation-failures true
mv tmp.jsonl ${PATH_TO_DATA}/synthetic_sentinel/math_dataset_synthetic/DuplicateOneStep_SwapOneStep_ExtrinsicHallucinatedStep_test.jsonl

for g in RandomNumber RandomOperation ShuffleNumbers ShuffleOperations
do
    python -m parlai.scripts.convert_data_to_json_format -t math_dataset --step-by-step-style none --step-perturbations SwapOneStep DuplicateOneStep --math-step-perturbations $g -dt test --skip-perturbation-failures true
    mv tmp.jsonl ${PATH_TO_DATA}/synthetic_sentinel/math_dataset_synthetic/DuplicateOneStep_SwapOneStep_${g}_test.jsonl
done

echo "Restoring partial perturbations ..."
# ==== restore partial perturbations for synthetic datasets
mkdir -p ${PATH_TO_DATA}/synthetic_50%
python ${RESTORE_SCRIPT} --dataset-path ${PATH_TO_DATA}/synthetic --perturbation-ids ${PATH_TO_DATA}/unperturbed_ids.json --percentage 50 --out-dir ${PATH_TO_DATA}/synthetic_50%

mkdir -p ${PATH_TO_DATA}/synthetic_sentinel
python ${RESTORE_SCRIPT} --dataset-path ${PATH_TO_DATA}/synthetic_sentinel --perturbation-ids ${PATH_TO_DATA}/unperturbed_ids.json --percentage 50 --out-dir ${PATH_TO_DATA}/synthetic_sentinel_50%
