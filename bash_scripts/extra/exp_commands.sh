
# verify that these two lead to same results. note differences in training time. 
# python bart_submit_ft_multiwoz.py -i pft -v 2.1 -b 8 -u 1 -f # notes:  
# python bart_submit_ft_multiwoz.py -i pft -v 2.1 -b 4 -u 2 -f #


# test out results with different batch sizes. 
# version=2.3
# for init_config in scratch pft muppet; do 
# # for init_config in muppet; do 
#     python bart_submit_ft_multiwoz.py -i $init_config -v $version -b 4 -u 1 #
#     # python bart_submit_ft_multiwoz.py -i $init_config -v $version -b 8 -u 2 #
#     python bart_submit_ft_multiwoz.py -i $init_config -v $version -b 8 -u 1 #

#     python bart_submit_ft_multiwoz.py -i $init_config -v $version -b 4 -u 1 -f #
#     # python bart_submit_ft_multiwoz.py -i $init_config -v $version -b 8 -u 2 -f #
#     python bart_submit_ft_multiwoz.py -i $init_config -v $version -b 8 -u 1 -f #
# done 

# version=2.1
# for init_config in scratch pft muppet; do 
# # for init_config in muppet; do 
#     python bart_submit_ft_multiwoz.py -i $init_config -v $version -b 4 -u 1 #
#     # python bart_submit_ft_multiwoz.py -i $init_config -v $version -b 8 -u 2 # this config never led to best results
#     python bart_submit_ft_multiwoz.py -i $init_config -v $version -b 8 -u 1 #

#     python bart_submit_ft_multiwoz.py -i $init_config -v $version -b 4 -u 1 -f #
#     # python bart_submit_ft_multiwoz.py -i $init_config -v $version -b 8 -u 2 -f #
#     python bart_submit_ft_multiwoz.py -i $init_config -v $version -b 8 -u 1 -f #
# done 


# rerun subset of experiments for saving all intermediate checkpoints 
# based on results stored at https://docs.google.com/spreadsheets/d/1OIbKI4LYIAcOeb9nwgLA7zzG_y1hpAoeT785jn_kOfo/edit#gid=752636205

"""scratch"""
# python bart_submit_ft_multiwoz.py -i scratch -v 2.1 -b 8 -u 1 -lr 5e-5 #
# python bart_submit_ft_multiwoz.py -i scratch -v 2.1 -b 4 -u 1 -lr 5e-5 -f #

# # # test 
# python bart_submit_ft_multiwoz.py -i scratch -v 2.3 -b 4 -u 1 -lr 5e-5 -t True #

# python bart_submit_ft_multiwoz.py -i scratch -v 2.3 -b 4 -u 1 -lr 5e-5 #
# python bart_submit_ft_multiwoz.py -i scratch -v 2.3 -b 4 -u 1 -lr 5e-5 -f #

"""pft"""
# python bart_submit_ft_multiwoz.py -i pft -v 2.1 -b 8 -u 2 -lr 1e-5 #
# python bart_submit_ft_multiwoz.py -i pft -v 2.1 -b 4 -u 2 -lr 5e-5 -f #

# python bart_submit_ft_multiwoz.py -i pft -v 2.3 -b 4 -u 1 -lr 5e-5 #
# python bart_submit_ft_multiwoz.py -i pft -v 2.3 -b 4 -u 2 -lr 1e-5 -f #

"""muppet"""
# python bart_submit_ft_multiwoz.py -i muppet -v 2.1 -b 8 -u 1 -lr 1e-4 #
# python bart_submit_ft_multiwoz.py -i muppet -v 2.1 -b 4 -u 1 -lr 1e-4 -f #

# python bart_submit_ft_multiwoz.py -i muppet -v 2.3 -b 4 -u 1 -lr 1e-4 #
# python bart_submit_ft_multiwoz.py -i muppet -v 2.3 -b 4 -u 1 -lr 5e-5 -f #


"""granular training ()"""
# python bart_submit_ft_multiwoz.py -i muppet -v 2.3 -b 4 -u 1 -lr 1e-4 #
# python bart_submit_ft_multiwoz.py -i scratch -v 2.3 -b 4 -u 1 -lr 5e-5 #
# python bart_submit_ft_multiwoz.py -i pft -v 2.3 -b 4 -u 1 -lr 5e-5 #

"""for SOLOIST-BART """
# python bart_submit_ft_multiwoz.py -i pft -v 2.3 -b 4 -u 1 -lr 5e-5 #
# python bart_submit_ft_multiwoz.py -i pft -v 2.3 -b 4 -u 1 -lr 1e-5 --no_prompt -f #

"""for SimpleTOD """
# python bart_submit_ft_multiwoz.py -i simpletod -v 2.3 -b 4 -u 1 -lr 5e-5 -m gpt2 --no_prompt
# python bart_submit_ft_multiwoz.py -i simpletod -v 2.3 -b 4 -u 1 -lr 5e-5 -m gpt2 --no_prompt -f

""" SGD pretraining for gpt2 """  # make sure to check the tasks in bart_submit_pft is google_sgd 
# python bart_submit_pft.py -m gpt2 -lr 5e-6 


"""for SOLOIST (GPT-2)"""
python bart_submit_ft_multiwoz.py -i soloist -v 2.3 -b 4 -u 1 -lr 5e-5 -m gpt2 --no_prompt 
python bart_submit_ft_multiwoz.py -i soloist -v 2.3 -b 4 -u 1 -lr 5e-5 -m gpt2 --no_prompt -f 