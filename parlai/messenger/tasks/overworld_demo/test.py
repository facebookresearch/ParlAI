
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task


def main():
    opt = {}
    opt["datapath"]="/Users/jase/src/ParlAI/data"
    opt["model"]="/Users/jase/src/ParlAI/parlai_external.projects.personachat.memnn1hop.memnn1hop:Memnn1hopAgent"
    opt["model_file"]="/Users/jase/src/ParlAI/parlai_external/data/personachat/memnn2hop_sweep/persona-none_rephraseTrn-True_rephraseTst-False_lr-0.1_esz-500_margin-0.1_tfidf-False_shareEmb-True_hops0_lins0/model"
    opt['fixed_candidates_file'] = "/Users/jase/src/ParlAI/data/personachat/cands.txt"
    model = create_agent(opt)
    a = { 'text': 'i like cats', 'episode_done': False}
    model.observe(a)
    response = model.act()
    print(response)

main()
