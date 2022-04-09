checkpointfolder="/some/storage"

# freebeam="freebeam_"
freebeam=""

for target in unforced forced_IDK forced_TRY forced_YEA; do
    parlai eval -dt train:evalmode \
        -t parlai.projects.metacognition.agents:CertaintyOntoTriviaQATeacher \
        --simplify-certainty False --classes '<IDK>' '<TRY>' '<YEA>' '<EVA>' \
        --model bert_classifier --save-world-logs True \
        --report-filename ${checkpointfolder}/ParlAITriviaQA/triviaqa_full_166_${freebeam}finetuned_alcest_94c_${target} \
        -mf ${checkpointfolder}/202009*/sweep_mccc_full_bigthief/166/model \
        --triviaqa-run-to-be-annotated ${checkpointfolder}/ParlAITriviaQA/NoEvidenceUnion_blender_3B_trainset_${freebeam}finetuned_alcest_94c_${target}_*_replies.jsonl
done
