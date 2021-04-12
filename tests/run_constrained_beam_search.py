from parlai.scripts.interactive import Interactive

if __name__ == '__main__':
        # call it with particular args
        Interactive.main(
            model='transformer/generator',
            task='blended_skill_talk',
            include_personas=False,
            include_initial_utterances=False,

            #history_size=2,
            beam_size=4,
            beam_min_length=20,
            beam_context_block_ngram=3,
            beam_block_ngram=3,
            inference='constrainedbeam',
            constraints=["a feeling of attraction"],
            #inference='beam',
            model_parallel=True,
            # the model_file is a filename path pointing to a particular model dump.
            # Model files that begin with "zoo:" are special files distributed by the ParlAI team.
            # They'll be automatically downloaded when you ask to use them.
            #model_file = "zoo:blender/blender_3B/model"
            model_file="zoo:blender/blender_90M/model"
        )



