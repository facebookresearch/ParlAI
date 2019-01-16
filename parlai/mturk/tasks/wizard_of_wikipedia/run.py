# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.agents import create_task_agent_from_taskname, create_agent
from parlai.core.params import ParlaiParser
from parlai.core.utils import AttrDict
from parlai.mturk.core.mturk_manager import MTurkManager
from worlds import \
    MTurkWizardOfWikipediaWorld, RoleOnboardWorld, PersonasGenerator, \
    WIZARD, APPRENTICE
from task_config import task_config
from parlai.core.dict import DictionaryAgent
import os
import copy
import tqdm
import pickle
import parlai.core.build_data as build_data
from urllib.parse import unquote


def setup_retriever(opt):
    print('[ Setting up Retriever ]')
    task = 'wikipedia:full'
    ret_opt = copy.deepcopy(opt)
    ret_opt['model_file'] = 'models:wikipedia_full/tfidf_retriever/model'
    ret_opt['retriever_num_retrieved'] = opt.get('num_passages_retrieved', 7)
    ret_opt['retriever_mode'] = 'keys'
    ret_opt['override'] = {'remove_title': False}
    ir_agent = create_agent(ret_opt)
    return ir_agent, task


def setup_title_to_passage(opt):
    print('[ Setting up Title to Passage Dict ]')
    saved_dp = os.path.join(os.getcwd() + '/data/', 'title_to_passage.pkl')
    if os.path.exists(saved_dp):
        print('[ Loading from saved location, {} ]'.format(saved_dp))
        with open(saved_dp, 'rb') as f:
            title_to_passage = pickle.load(f)
            return title_to_passage
    topics_path = '{}/personas_with_wiki_links.txt'.format(os.getcwd())
    topics = []
    with open(topics_path) as f:
        text = f.read()
        personas = text.split('\n\n')
        for persona in personas:
            persona = persona.split('\n')
            for i in range(1, len(persona)):
                p_i = persona[i]
                if 'https' in p_i:
                    topic = unquote(p_i[p_i.rfind('/') + 1:]).replace('_',
                                                                      ' ')
                    topics.append(topic)
    ordered_opt = opt.copy()
    ordered_opt['datatype'] = 'train:ordered:stream'
    ordered_opt['batchsize'] = 1
    ordered_opt['numthreads'] = 1
    ordered_opt['task'] = 'wikipedia:full:key-value'
    teacher = create_task_agent_from_taskname(ordered_opt)[0]
    title_to_passage = {}
    i = 0
    length = teacher.num_episodes()
    pbar = tqdm.tqdm(total=length)
    while not teacher.epoch_done():
        pbar.update(1)
        i += 1
        action = teacher.act()
        title = action['text']
        if title in topics:
            text = action['labels'][0]
            title_to_passage[title] = text
    pbar.close()
    print('[ Finished Building Title to Passage dict; saving now]')
    with open(saved_dp, 'wb') as f:
        pickle.dump(title_to_passage, f)
    return title_to_passage


def setup_personas_with_wiki_links(opt):
    fname = 'personas_with_wiki_links.txt'
    file_path = '{}/{}'.format(os.getcwd(), fname)
    if not os.path.exists(file_path):
        url = 'http://parl.ai/downloads/wizard_of_wikipedia/' + fname
        build_data.download(url, os.getcwd(), fname)


def main():
    """
        Wizard of Wikipedia Data Collection Task.

        The task involves two people holding a conversation. One dialog partner
        chooses a topic to discuss, and then dialog proceeds.

        One partner is the Wizard, who has access to retrieved external
        information conditioned on the last two utterances, as well as
        information regarding the chosen topic.

        The other partner is the Apprentice, who assumes the role of someone
        eager to learn about the chosen topic.
    """
    argparser = ParlaiParser(False, False)
    DictionaryAgent.add_cmdline_args(argparser)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.add_argument('-min_t', '--min_turns', default=3, type=int,
                           help='minimum number of turns')
    argparser.add_argument('-max_t', '--max_turns', default=5, type=int,
                           help='maximal number of chat turns')
    argparser.add_argument('-mx_rsp_time', '--max_resp_time', default=120,
                           type=int,
                           help='time limit for entering a dialog message')
    argparser.add_argument('-mx_onb_time', '--max_onboard_time', type=int,
                           default=300, help='time limit for turker'
                           'in onboarding')
    argparser.add_argument('--persona-type', default='both', type=str,
                           choices=['both', 'self', 'other'],
                           help='Which personas to load from personachat')
    argparser.add_argument('--auto-approve-delay', type=int,
                           default=3600 * 24 * 1, help='how long to wait for  \
                           auto approval')
    argparser.add_argument('--word-overlap-threshold', type=int, default=2,
                           help='How much word overlap we want between message \
                           and checked sentence')
    argparser.add_argument('--num-good-sentence-threshold', type=int, default=2,
                           help='How many good sentences with sufficient overlap \
                           are necessary for turker to be considered good.')
    argparser.add_argument('--num-passages-retrieved', type=int, default=7,
                           help='How many passages to retrieve per dialog \
                           message')

    opt = argparser.parse_args()
    directory_path = os.path.dirname(os.path.abspath(__file__))
    opt['task'] = os.path.basename(directory_path)
    if 'data_path' not in opt:
        opt['data_path'] = os.getcwd() + '/data/' + opt['task']
        opt['current_working_dir'] = os.getcwd()
    opt.update(task_config)

    mturk_agent_ids = [APPRENTICE, WIZARD]
    opt['min_messages'] = 2

    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids=mturk_agent_ids
    )
    setup_personas_with_wiki_links(opt)
    ir_agent, task = setup_retriever(opt)
    persona_generator = PersonasGenerator(opt)
    wiki_title_to_passage = setup_title_to_passage(opt)
    mturk_manager.setup_server(task_directory_path=directory_path)
    worker_roles = {}
    connect_counter = AttrDict(value=0)

    try:
        mturk_manager.start_new_run()
        if not opt['is_sandbox']:
            with open(os.path.join(opt['current_working_dir'], 'mtdont.txt')) as f:
                lines = [l.replace('\n', '') for l in f.readlines()]
                for w in lines:
                    mturk_manager.soft_block_worker(w)

        def run_onboard(worker):
            role = mturk_agent_ids[connect_counter.value % len(mturk_agent_ids)]
            connect_counter.value += 1
            worker_roles[worker.worker_id] = role
            worker.persona_generator = persona_generator
            world = RoleOnboardWorld(opt, worker, role)
            world.parley()
            world.shutdown()

        mturk_manager.set_onboard_function(onboard_function=run_onboard)
        mturk_manager.ready_to_accept_workers()
        mturk_manager.create_hits()

        def check_workers_eligibility(workers):
            if opt['is_sandbox']:
                return workers
            valid_workers = {}
            for worker in workers:
                worker_id = worker.worker_id
                if worker_id not in worker_roles:
                    '''Something went wrong...'''
                    continue
                role = worker_roles[worker_id]
                if role not in valid_workers:
                    valid_workers[role] = worker
                if len(valid_workers) == 2:
                    break
            return valid_workers.values() if len(valid_workers) == 2 else []

        eligibility_function = {
            'func': check_workers_eligibility,
            'multiple': True,
        }

        def assign_worker_roles(workers):
            if opt['is_sandbox']:
                for i, worker in enumerate(workers):
                    worker.id = mturk_agent_ids[i % len(mturk_agent_ids)]
            else:
                for worker in workers:
                    worker.id = worker_roles[worker.worker_id]

        def run_conversation(mturk_manager, opt, workers):
            agents = workers[:]
            if not opt['is_sandbox']:
                for agent in agents:
                    worker_roles.pop(agent.worker_id)
            conv_idx = mturk_manager.conversation_index
            world = MTurkWizardOfWikipediaWorld(
                opt,
                agents=agents,
                world_tag='conversation t_{}'.format(conv_idx),
                ir_agent=ir_agent,
                wiki_title_to_passage=wiki_title_to_passage,
                task=task
            )
            world.reset_random()
            while not world.episode_done():
                world.parley()
            world.save_data()
            if (world.convo_finished and
                    not world.good_wiz and
                    not opt['is_sandbox']):
                mturk_manager.soft_block_worker(world.wizard_worker)
            world.shutdown()
            world.review_work()

        mturk_manager.start_task(
            eligibility_function=eligibility_function,
            assign_role_function=assign_worker_roles,
            task_function=run_conversation
        )

    except BaseException:
        raise
    finally:
        mturk_manager.expire_all_unassigned_hits()
        mturk_manager.shutdown()


if __name__ == '__main__':
    main()
