from argparse import ArgumentParser

def setup_args():
    argparser = ArgumentParser()
    argparser.add_argument('-if', '--infile', type=str, 
        default='/private/home/bhancock/ParlAI/data/ConvAI2/train_self_original.txt')
    argparser.add_argument('-of', '--outfile', type=str,
        default='/private/home/bhancock/ParlAI/parlai_internal/projects/metadialog/convai2_cands.txt')
    config = vars(argparser.parse_args())
    return config

def main(config):
    with open(config['infile'], 'r') as fin, open(config['outfile'], 'w') as fout:
        for line in fin.readlines():
            if 'persona' in line:
                continue
            first_space = line.index(' ')
            first_tab = line.index('\t')
            candidate = line[first_space+1 : first_tab]
            fout.write(candidate + '\n')

if __name__ == '__main__':
    config = setup_args()
    main(config)
