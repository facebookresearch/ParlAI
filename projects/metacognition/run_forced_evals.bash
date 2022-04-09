#!/bin/bash

sweepfolder="/some/storage/sweep_mccfts2_series"
outputdir="/some/storage"

header () {
echo "#!/bin/bash"
echo "#SBATCH --output=${sweepfolder}/slurm_logs/slrm_stdout.%j"
echo "#SBATCH --error=${sweepfolder}/slurm_logs/slrm_stderr.%j"
cat <<'END_HEREDOC'
cd /some/storage/ParlAI/
export PARLAI_DATAPATH=/some/storage/ParlAI/data
export PARLAI_DOWNPATH=/some/storage/ParlAI/downloads
. /some/path/anaconda3/5.0.1/etc/profile.d/conda.sh
conda activate conda_parlai
export PYTHONPATH=/some/storage/ParlAI:$PYTHONPATH
END_HEREDOC
}

footer () {
cat <<'END_HEREDOC'
CHILD="$!"
wait "$CHILD"
RETVAL=$?
sleep 30
exit $RETVAL
END_HEREDOC
}

# Normal runs of the valid set

for run in ${sweepfolder}/???; do
    run="${run: -3}"

    header > "${sweepfolder}/eval_${run}_unforced.bash"
    echo "parlai eval -t parlai.projects.metacognition.agents:NoEvidenceUnionUnforcedTeacher -dt valid --force-same True --save-world-logs True --report-filename ${outputdir}/NoEvidenceUnion_blender_3B_finetuned_${sweepfolder##*_}_${run}_unforced -mf ${sweepfolder}/${run}/model --fp16 True --truncate 128 --fp16-impl mem_efficient -bs 12 -m parlai.projects.metacognition.agents:PartialLossExtractingTransformerGeneratorAgent --model-parallel True --hf-skip-special-tokens False --special-tok-lst '<IDK>,<TRY>,<YEA>,<SAME>,<DIFF>'" >> "${sweepfolder}/eval_${run}_unforced.bash"
    footer >> "${sweepfolder}/eval_${run}_unforced.bash"
    sbatch ${sweepfolder}/eval_${run}_unforced.bash


    for force in IDK TRY YEA; do
        header > "${sweepfolder}/eval_${run}_${force}.bash"
        echo "parlai eval -t parlai.projects.metacognition.agents:NoEvidenceUnionForced${force}Teacher -dt valid --force-same True --save-world-logs True --report-filename ${outputdir}/NoEvidenceUnion_blender_3B_finetuned_${sweepfolder##*_}_${run}_forced_${force} -mf ${sweepfolder}/${run}/model --fp16 True --truncate 128 --fp16-impl mem_efficient -bs 12 -m parlai.projects.metacognition.agents:PartialLossExtractingTransformerGeneratorAgent --model-parallel True --hf-skip-special-tokens False --special-tok-lst '<IDK>,<TRY>,<YEA>,<SAME>,<DIFF>'" >> "${sweepfolder}/eval_${run}_${force}.bash"
        footer >> "${sweepfolder}/eval_${run}_${force}.bash"
        sbatch ${sweepfolder}/eval_${run}_${force}.bash
    done
done


## Runs of the training set (necessary for stage1-2 tuning)

# for run in /some/storage/20201110/sweep_mccfts2_alcest/94c; do
#     run="${run: -3}"

#     header > "${sweepfolder}/eval_train_${run}_unforced.bash"
#     echo "parlai eval -t triviaqa:NoEvidenceUnion -dt train:evalmode --save-world-logs True --report-filename /some/storage/ParlAITriviaQA/NoEvidenceUnion_blender_3B_trainset_finetuned_${sweepfolder##*_}_${run}_unforced -mf ${sweepfolder}/${run}/model --fp16 True --truncate 128 --fp16-impl mem_efficient -bs 12 -m parlai.projects.metacognition.agents:PartialLossExtractingTransformerGeneratorAgent --model-parallel True --hf-skip-special-tokens False --special-tok-lst '<IDK>,<TRY>,<YEA>,<SAME>,<DIFF>'" >> "${sweepfolder}/eval_train_${run}_unforced.bash"
#     footer >> "${sweepfolder}/eval_train_${run}_unforced.bash"

#     for force in IDK TRY YEA; do
#         header > "${sweepfolder}/eval_train_${run}_${force}.bash"
#         echo "parlai eval -t parlai.projects.metacognition.agents:NoEvidenceUnionForced${force}Teacher -dt train:evalmode --save-world-logs True --report-filename /some/storage/ParlAITriviaQA/NoEvidenceUnion_blender_3B_trainset_finetuned_${sweepfolder##*_}_${run}_forced_${force} -mf ${sweepfolder}/${run}/model --fp16 True --truncate 128 --fp16-impl mem_efficient -bs 12 -m parlai.projects.metacognition.agents:PartialLossExtractingTransformerGeneratorAgent --model-parallel True --hf-skip-special-tokens False --special-tok-lst '<IDK>,<TRY>,<YEA>,<SAME>,<DIFF>'" >> "${sweepfolder}/eval_train_${run}_${force}.bash"
#         footer >> "${sweepfolder}/eval_train_${run}_${force}.bash"
#     done
# done


# for run in /some/storage/20201110/sweep_mccfts2_alcest/94c; do
#     run="${run: -3}"

#     header > "${sweepfolder}/eval_train_freebeam_${run}_unforced.bash"
#     echo "parlai eval -t triviaqa:NoEvidenceUnion -dt train:evalmode --save-world-logs True --report-filename /some/storage/ParlAITriviaQA/NoEvidenceUnion_blender_3B_trainset_freebeam_finetuned_${sweepfolder##*_}_${run}_unforced -mf ${sweepfolder}/${run}/model --fp16 True --truncate 128 --fp16-impl mem_efficient -bs 12 -m parlai.projects.metacognition.agents:PartialLossExtractingTransformerGeneratorAgent --model-parallel True --hf-skip-special-tokens False --special-tok-lst '<IDK>,<TRY>,<YEA>,<SAME>,<DIFF>' --beam-context-block-ngram 0 --beam-block-ngram 0 --beam-min-length 1" >> "${sweepfolder}/eval_train_freebeam_${run}_unforced.bash"
#     footer >> "${sweepfolder}/eval_train_freebeam_${run}_unforced.bash"

#     for force in IDK TRY YEA; do
#         header > "${sweepfolder}/eval_train_freebeam_${run}_${force}.bash"
#         echo "parlai eval -t parlai.projects.metacognition.agents:NoEvidenceUnionForced${force}Teacher -dt train:evalmode --save-world-logs True --report-filename /some/storage/ParlAITriviaQA/NoEvidenceUnion_blender_3B_trainset_freebeam_finetuned_${sweepfolder##*_}_${run}_forced_${force} -mf ${sweepfolder}/${run}/model --fp16 True --truncate 128 --fp16-impl mem_efficient -bs 12 -m parlai.projects.metacognition.agents:PartialLossExtractingTransformerGeneratorAgent --model-parallel True --hf-skip-special-tokens False --special-tok-lst '<IDK>,<TRY>,<YEA>,<SAME>,<DIFF>' --beam-context-block-ngram 0 --beam-block-ngram 0 --beam-min-length 1" >> "${sweepfolder}/eval_train_freebeam_${run}_${force}.bash"
#         footer >> "${sweepfolder}/eval_train_freebeam_${run}_${force}.bash"
#     done
# done
