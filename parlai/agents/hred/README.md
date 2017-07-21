### Description
This repository hosts the Latent Variable Hierarchical Recurrent Encoder-Decoder RNN model with Gaussian and piecewise constant latent variables for generative dialog modeling, as well as the HRED baseline model. These models were proposed in the paper "Piecewise Latent Variables for Neural Variational Text Processing" by Serban et al.


### Truncated BPTT
All models are implemented using Truncated Backpropagation Through Time (Truncated BPTT).
The truncated computation is carried out by splitting each document (dialogue) into shorter sequences (e.g. 80 tokens) and computing gradients for each sequence separately, such that the hidden state of the RNNs on each subsequence is initialized from the preceding sequence (i.e. the hidden states have been forward propagated through the previous states).


### Creating Datasets
The script convert-text2dict.py can be used to generate model datasets based on text files with dialogues.
It only requires that the document contains end-of-utterance tokens &lt;/s&gt; which are used to construct the model graph, since the utterance encoder is only connected to the dialogue encoder at the end of each utterance.

Prepare your dataset as a text file for with one document per line (e.g. one dialogue per line). The documents are assumed to be tokenized. If you have validation and test sets, they must satisfy the same requirements.

Once you're ready, you can create the model dataset files by running:

python convert-text2dict.py &lt;training_file&gt; --cutoff &lt;vocabulary_size&gt; Training
python convert-text2dict.py &lt;validation_file&gt; --dict=Training.dict.pkl Validation
python convert-text2dict.py &lt;test_file&gt; --dict=Training.dict.pkl &lt;vocabulary_size&gt; Test

where &lt;training_file&gt;, &lt;validation_file&gt; and &lt;test_file&gt; are the training, validation and test files, and &lt;vocabulary_size&gt; is the number of tokens that you want to train on (all other tokens, but the most frequent &lt;vocabulary_size&gt; tokens, will be converted to &lt;unk&gt; symbols).

NOTE: The script automatically adds the following special tokens specific to movie script dialogues:
- end-of-utterance: &lt;/s&gt;
- end-of-dialogue: &lt;/d&gt;
- first speaker: &lt;first_speaker&gt;
- second speaker: &lt;second_speaker&gt;
- third speaker: &lt;third_speaker&gt;
- minor speaker: &lt;minor_speaker&gt;
- voice over: &lt;voice_over&gt;
- off screen: &lt;off_screen&gt;
- pause: &lt;pause&gt;

If these do not exist in your dataset, you can safely ignore these. The model will learn to assign approximately zero probability mass to them.


### Model Training
If you have Theano with GPU installed (bleeding edge version), you can train the model as follows:
1) Clone the Github repository
2) Unpack your dataset files into "Data" directory.
3) Create a new prototype inside state.py (look at prototype_test_variational for an example)
4) From the terminal, cd into the code directory and run:

    THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32 python train.py --prototype <prototype_name> > Model_Output.txt

where &lt;prototype_name&gt; is a state (model configuration/architecture) defined inside state.py.
Training a model to convergence on a modern GPU on the Ubuntu Dialogue Corpus with 46 million tokens takes about 2 weeks. If your GPU runs out of memory, you can adjust the batch size (bs) parameter in the model state, but training will be slower. You can also play around with the other parameters inside state.py.


### Model Sampling & Testing
To generate model responses using beam search run:

    THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=cuda python sample.py <model_name> <contexts> <model_outputs> --beam_search --n-samples=<beams> --ignore-unk --verbose

where &lt;model_name&gt; is the name automatically generated during training, &lt;contexts&gt; is a file containing the dialogue contexts with one dialogue per line, and &lt;beams&gt; is the size of the beam search. The results are saved in the file &lt;model_outputs&gt;.


### Citation
If you build on this work, we'd really appreciate it if you could cite our papers:

    Piecewise Latent Variables for Neural Variational Text Processing. Iulian V. Serban, Alexander G. Ororbia II, Joelle Pineau, Aaron Courville, Yoshua Bengio. 2017. https://arxiv.org/abs/1612.00377

    A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues. Iulian V. Serban, Alessandro Sordoni, Ryan Lowe, Laurent Charlin, Joelle Pineau, Aaron Courville, Yoshua Bengio. 2016. http://arxiv.org/abs/1605.06069

    Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models. Iulian V. Serban, Alessandro Sordoni, Yoshua Bengio, Aaron Courville, Joelle Pineau. 2016. AAAI. http://arxiv.org/abs/1507.04808.


### Reproducing Results in "Piecewise Latent Variables for Neural Variational Text Processing" 
The results reported in the paper "Piecewise Latent Variables for Neural Variational Text Processing" by Serban et al. are based on the following model states found inside state.py:

    prototype_ubuntu_GaussPiecewise_NormOp_VHRED_Baseline_Exp1 (HRED baseline)
    prototype_ubuntu_GaussPiecewise_NormOp_VHRED_Exp5 (P-VHRED)
    prototype_ubuntu_GaussPiecewise_NormOp_VHRED_Exp7 (G-VHRED)
    prototype_ubuntu_GaussPiecewise_NormOp_VHRED_Exp9 (H-VHRED)

To reproduce these results from scratch, you must follow these steps:

1) Download and unpack the preprocessed Ubuntu dataset available from http://www.iulianserban.com/Files/UbuntuDialogueCorpus.zip.

2) a) Clone this Github repository locally on a machine. Use a machine with a fast GPU with large memory (preferably 12GB).

   b) Reconfigure the model states above in state.py appropriately:
      1) Change 'train\_dialogues', 'valid\_dialogues', 'test\_dialogues' to the path for the Ubuntu dataset files.
      2) Change 'dictionary' to the path for the dictionary.

   c) Train up the model. This takes about 2 weeks time!
      For example, for "prototype\_ubuntu\_GaussPiecewise\_NormOp\_VHRED\_Exp9" run:

        THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32 python train.py --prototype prototype_ubuntu_GaussPiecewise_NormOp_VHRED_Exp9 &> Model_Output.txt

      The model will be saved inside the directory Output/.
      If the machine runs out of GPU memory, reduce the batch size (bs) and maximum number of gradient steps (max_grad_steps) in the model state.

   d) Generate outputs using beam search with size 5 on the Ubuntu test set.
      To do this, run:

        THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32 python sample.py <model_path_prefix> <text_set_contexts> <output_file> --beam_search --n-samples=5 --n-turns=1 --verbose

      where &lt;model_path_prefix&gt; is the path to the saved model parameters excluding the postfix (e.g. Output/1482712210.89_UbuntuModel),
      &lt;text_set_contexts&gt; is the path to the Ubuntu test set contexts and  &lt;output_file&gt; is where the beam outputs will be stored.

   e) Compute performance using activity- and entity-based metrics.
      Follow the instructions given here: https://github.com/julianser/Ubuntu-Multiresolution-Tools.


Following all steps to reproduce the results requires a few weeks time and, depending on your setup, may also require changing your Theano configuraiton and the state file. Therefore, we have also made available the trained models and the generated model responses on the test set.

You can find the trained models here: https://drive.google.com/open?id=0B06gib_77EnxaDg2VkV1N1huUjg.

You can find the model responses generated using beam search in this repository inside "TestSet_BeamSearch_Outputs/".


### Datasets
The pre-processed Ubuntu Dialogue Corpus and model responses used are available at: http://www.iulianserban.com/Files/UbuntuDialogueCorpus.zip.

The original Ubuntu Dialogue Corpus as released by Lowe et al. (2015) can be found here: http://cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0/

Unfortunately due to Twitter's terms of service we are not allowed to distribute Twitter content. Therefore we can only make available the tweet IDs, which can then be used with the Twitter API to build a similar dataset. The tweet IDs and model test responses can be found here: http://www.iulianserban.com/Files/TwitterDialogueCorpus.zip.

### References

    Piecewise Latent Variables for Neural Variational Text Processing. Iulian V. Serban, Alexander G. Ororbia II, Joelle Pineau, Aaron Courville, Yoshua Bengio. 2017. https://arxiv.org/abs/1612.00377

    A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues. Iulian Vlad Serban, Alessandro Sordoni, Ryan Lowe, Laurent Charlin, Joelle Pineau, Aaron Courville, Yoshua Bengio. 2016a. http://arxiv.org/abs/1605.06069

    Multiresolution Recurrent Neural Networks: An Application to Dialogue Response Generation. Iulian Vlad Serban, Tim Klinger, Gerald Tesauro, Kartik Talamadupula, Bowen Zhou, Yoshua Bengio, Aaron Courville. 2016b. http://arxiv.org/abs/1606.00776.

    Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models. Iulian V. Serban, Alessandro Sordoni, Yoshua Bengio, Aaron Courville, Joelle Pineau. 2016c. AAAI. http://arxiv.org/abs/1507.04808.

    Training End-to-End Dialogue Systems with the Ubuntu Dialogue Corpus. Ryan Lowe, Nissan Pow, Iulian V. Serban, Laurent Charlin, Chia-Wei Liu, Joelle Pineau. 2017. Dialogue & Discourse Journal. http://www.cs.mcgill.ca/~jpineau/files/lowe-dialoguediscourse-2017.pdf

    The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems. Ryan Lowe, Nissan Pow, Iulian Serban, Joelle Pineau. 2015. SIGDIAL. http://arxiv.org/abs/1506.08909.
