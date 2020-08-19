#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import io
import os
import json
from parlai.utils.misc import msg_to_str

rand = random.Random(42)


# TODO Can we grab these from somewhere in the built content?
CURRENT_HOBBOT_MODELS = [
    'orig_light_poly',
    'orig_light_poly_boring',
    'reddit_polyranker',
    'sw12_bigmlm_smallcode5',
    'sw17_bigmlm_smallcode5',
    'light_large_with_neg?block_ngrams=0,boring_alpha=-10,fixed_candidates_path=filtered_light_and_gen_cands.txt,from_speaker_bonus=50,normalized=none,to_speaker_bonus=50',
    'light_large_with_neg?block_ngrams=0,boring_alpha=-10,fixed_candidates_path=filtered_light_and_wild_cands.txt,from_speaker_bonus=50,normalized=none,to_speaker_bonus=50',
    'light_large_with_neg?block_ngrams=0,boring_alpha=-10,fixed_candidates_path=filtered_light_cands.txt,from_speaker_bonus=50,normalized=none,to_speaker_bonus=50',
    'light_large_with_neg?block_ngrams=0,boring_alpha=0,fixed_candidates_path=filtered_light_and_wild_cands.txt,from_speaker_bonus=0,normalized=none,to_speaker_bonus=0',
    'light_large_with_neg?block_ngrams=0,boring_alpha=0,fixed_candidates_path=filtered_light_cands.txt,from_speaker_bonus=0,normalized=none,to_speaker_bonus=0',
    'light_large_with_neg?block_ngrams=0,boring_alpha=0,fixed_candidates_path=filtered_light_cands.txt,from_speaker_bonus=50,normalized=none,to_speaker_bonus=50',
    'light_large_with_neg?block_ngrams=3,boring_alpha=-10,fixed_candidates_path=filtered_light_cands.txt,from_speaker_bonus=50,normalized=none,to_speaker_bonus=50',
    'light_large_with_wild_and_neg?block_ngrams=0,boring_alpha=-10,fixed_candidates_path=filtered_light_and_gen_cands.txt,from_speaker_bonus=50,normalized=none,to_speaker_bonus=50',
    'light_large_with_wild_and_neg?block_ngrams=0,boring_alpha=-10,fixed_candidates_path=filtered_light_and_wild_cands.txt,from_speaker_bonus=50,normalized=none,to_speaker_bonus=50',
    'light_large_with_wild_and_neg?block_ngrams=0,boring_alpha=-10,fixed_candidates_path=filtered_light_cands.txt,from_speaker_bonus=50,normalized=none,to_speaker_bonus=50',
    'light_large_with_wild_and_neg?block_ngrams=0,boring_alpha=0,fixed_candidates_path=filtered_light_and_wild_cands.txt,from_speaker_bonus=0,normalized=none,to_speaker_bonus=0',
    'light_large_with_wild_and_neg?block_ngrams=0,boring_alpha=0,fixed_candidates_path=filtered_light_cands.txt,from_speaker_bonus=0,normalized=none,to_speaker_bonus=0',
    'light_large_with_wild_and_neg?block_ngrams=0,boring_alpha=0,fixed_candidates_path=filtered_light_cands.txt,from_speaker_bonus=50,normalized=none,to_speaker_bonus=50',
    'light_large_with_wild_and_neg?block_ngrams=3,boring_alpha=-10,fixed_candidates_path=filtered_light_cands.txt,from_speaker_bonus=50,normalized=none,to_speaker_bonus=50',
    'light_segment_gen_small?block_ngrams=3,boring_alpha=0,fixed_candidates_path=filtered_light_cands_segmented.txt,normalized=none',
    'light_small_with_wild_and_neg?block_ngrams=0,boring_alpha=-50,fixed_candidates_path=filtered_light_and_wild_cands.txt,from_speaker_bonus=20,normalized=none,to_speaker_bonus=20',
    'light_small_with_wild_and_neg?block_ngrams=0,boring_alpha=-50,fixed_candidates_path=filtered_light_cands.txt,from_speaker_bonus=20,normalized=none,to_speaker_bonus=20',
    'light_small_with_wild_and_neg?block_ngrams=0,boring_alpha=0,fixed_candidates_path=filtered_light_and_wild_cands.txt,from_speaker_bonus=0,normalized=none,to_speaker_bonus=0',
    'light_small_with_wild_and_neg?block_ngrams=0,boring_alpha=0,fixed_candidates_path=filtered_light_cands.txt,from_speaker_bonus=0,normalized=none,to_speaker_bonus=0',
    'light_small_with_wild_and_neg?block_ngrams=3,boring_alpha=-50,fixed_candidates_path=filtered_light_cands.txt,from_speaker_bonus=20,normalized=none,to_speaker_bonus=20',
    'orig_light_poly?block_ngrams=0,boring_alpha=-50,encode_candidate_vecs=True,fixed_candidates_path=filtered_light_and_wild_cands.txt,from_speaker_bonus=20,normalized=none,to_speaker_bonus=20',
    'orig_light_poly?block_ngrams=0,boring_alpha=-50,encode_candidate_vecs=True,fixed_candidates_path=filtered_light_cands.txt,from_speaker_bonus=20,normalized=none,to_speaker_bonus=20',
    'orig_light_poly?block_ngrams=0,boring_alpha=-50,fixed_candidates_path=filtered_light_and_wild_cands.txt,from_speaker_bonus=20,normalized=none,to_speaker_bonus=20',
    'orig_light_poly?block_ngrams=0,boring_alpha=-50,fixed_candidates_path=filtered_light_cands.txt,from_speaker_bonus=20,normalized=none,to_speaker_bonus=20',
    'orig_light_poly?block_ngrams=0,boring_alpha=0,fixed_candidates_path=filtered_light_and_wild_cands.txt,from_speaker_bonus=0,normalized=none,to_speaker_bonus=0',
    'orig_light_poly?block_ngrams=0,boring_alpha=0,fixed_candidates_path=filtered_light_cands.txt,from_speaker_bonus=0,normalized=none,to_speaker_bonus=0',
]


# locations in the LIGHT test unseen set.
unseen = [
    "the lords treasury",
    "mess hall of the servants",
    "grand hall of the city",
    "the endless strait",
    "frozen outpost",
    "inside of a igloo",
    "eating area of cave",
    "research laboratory",
    "the fountain of youth",
    "the dark kings throne room",
    "the cabin of an ogre",
    "lower dungeon",
    "reaper's lair",
    "stonehenge",
    "toad hollow",
    "the dungeon",
    "geothermal valley",
    "cloud tavern",
    "harp store",
    "entrance to a cave",
    "ice cave",
    "entry to icy fortress",
    "swamp",
    "faeries den",
    "a creature cave",
    "cavern",
    "torture chanber",
    "the devils castle",
    "whipping chamber",
    "lava lake",
    "ghost monk quarters",
    "astral plane",
    "bridge in the forest",
    "fighting arena inside tourney grounds",
    "tool shed",
    "farmhouse",
    "dragon lift",
    "cloud nine bar",
    "observatory",
    "wizards dragon egg incubatory",
    "the moon observatory",
    "main square of the city in the clouds",
    "hidden cave",
    "living room",
    "the abyss",
    "den",
    "festival room",
    "the royal tent",
    "large ice cave",
    "secret magician's workshop",
    "wizard's lair",
    "echo hall",
    "inside of a troll cave",
    "devil's den",
    "withered gardens",
    "the pit of despair",
    "lord's castle",
    "dungeon",
    "viewing room",
    "goddess hollow",
    "haunted basement",
    "magic room",
    "the glade of mysteries",
    "underwater marketplace",
    "research lab",
    "dry dock room",
    "underwater aquapolis",
    "the marketplace",
    "neptune's throne room",
    "the thrones of death",
    "elf's room",
]

# Score cutoff for test and valid sets
LIGHT_TEST_CUTOFF = 9


def use_feat(opt, field, ttype):
    if 'all' in opt.get(field) or opt.get(field) == ttype:
        return True
    else:
        return False


mx = 0


def fix_labels(act, opt):
    labels = act.get('labels', act.get('eval_labels'))
    clip = int(opt.get('light_use_clip_cands', 1000))
    while len(act['label_candidates']) >= clip:
        act['label_candidates'].pop()
    is_label_cand = {}
    is_label_cand[labels] = False
    for c in act['label_candidates']:
        if c in is_label_cand:
            is_label_cand[c] = True
    for l, has in is_label_cand.items():
        if has is False:
            print("***ADDING A LABEL CAND****")
            act['label_candidates'].append(l)


def get_no_affordance_actions(in_room, carrying, other_carrying, other_name):
    all_actions = []
    # Drop "a" prefix from all
    in_room = [i[2:] for i in in_room]
    carrying = [i[2:] for i in carrying]
    other_carrying = [i[2:] for i in other_carrying]
    for obj in other_carrying:
        all_actions.append('steal {} from {}'.format(obj, other_name))
    for obj in in_room:
        all_actions.append('get {}'.format(obj))
    for obj in carrying:
        for f in ['drop {}', 'wield {}', 'wear {}', 'remove {}', 'eat {}', 'drink {}']:
            all_actions.append(f.format(obj))
        all_actions.append('give {} to {}'.format(obj, other_name))
        for obj2 in in_room:
            all_actions.append('put {} in {}'.format(obj, obj2))
        for obj2 in carrying:
            if obj2 == obj:
                continue
            all_actions.append('put {} in {}'.format(obj, obj2))
    return list(set(all_actions))


def gen_no_affordance_actions_for_dialogue(d):
    characters = [d['agents'][0]['name'], d['agents'][1]['name']]
    other_char = 1
    no_affordance_actions = []
    last_carry = []
    for idx in range(len(d['speech'])):
        char = characters[other_char]
        curr_carry = d['carrying'][idx] + d['wearing'][idx] + d['wielding'][idx]
        no_affordance_actions_turn = get_no_affordance_actions(
            d['room_objects'][idx], curr_carry, last_carry, char
        )
        no_affordance_actions_turn += d['available_actions'][idx]
        no_affordance_actions.append(list(set(no_affordance_actions_turn)))
        last_carry = curr_carry
        other_char = 1 - other_char
    d['no_affordance_actions'] = no_affordance_actions


def write_dialog(opt, fw, d, label_type, split):
    l = len(d['speech'])
    msgs = []
    text = ''
    score = d['score']
    model_name = d['model_name']
    did_continue = d['did_continue']

    d['which'] = []
    for i in range(0, len(d['emote'])):
        lab = 'none'
        if d['emote'][i] is not None:
            lab = 'emote'
        if d['action'][i] is not None:
            lab = 'action'
        d['which'].append(lab)

    if opt.get('light_use_taskname', True):
        text = '_task_' + label_type + '\n'
    if opt['light_use_setting']:
        text += (
            '_setting_name '
            + d['setting']['name']
            + ", "
            + d['setting']['category']
            + '\n'
        )
        text += '_setting_desc ' + d['setting']['description'] + '\n'
    if opt['light_use_person_names']:
        text += '_partner_name ' + d['partner_agent']['name'] + '\n'
        text += '_self_name ' + d['self_agent']['name'] + '\n'
    if use_feat(opt, 'light_use_persona', 'self'):
        text += '_self_persona ' + d['self_agent']['persona'] + '\n'
    if use_feat(opt, 'light_use_persona', 'other'):
        text += '_other_persona ' + d['partner_agent']['persona'] + '\n'
    for i in range(0, l, 2):
        if i < l - 1:
            if (
                use_feat(opt, 'light_use_speech', 'partner')
                and d['speech'][i] is not None
            ):
                if opt['light_use_speech_prefix']:
                    text += '_partner_say '
                elif opt['light_use_person_names_prefix']:
                    text += f"*{d['partner_agent']['name']}*: "
                text += str(d['speech'][i]) + '\n'
            if (
                use_feat(opt, 'light_use_action', 'partner')
                and d['action'][i] is not None
            ):
                text += '_partner_act ' + str(d['action'][i]) + '\n'
            if (
                use_feat(opt, 'light_use_emote', 'partner')
                and d['emote'][i] is not None
            ):
                text += '_partner_emote ' + str(d['emote'][i]) + '\n'
            if opt.get('light_use_repeat') == 'self_last':
                if i > 0:
                    text = str(d['speech'][i - 1])
                else:
                    text = 'nothing'
            if opt.get('light_use_repeat') == 'partner_last':
                text = str(d['speech'][i])
            if opt.get('light_use_repeat') == 'both_last':
                text = ''
                if i > 0:
                    text += str(d['speech'][i - 1]) + ' '
                text += str(d['speech'][i])

            label = d[label_type][i + 1]
            if opt['light_use_person_names_prefix']:
                label = f"*{d['self_agent']['name']}*: {label}"
            used_current = False
            shown = {}
            if (
                use_feat(opt, 'light_use_current_self_output', 'speech')
                and label_type != 'speech'
                and d['speech'][i + 1] is not None
            ):
                if 'remove' not in opt['light_use_current_self_output']:
                    if opt['light_use_speech_prefix']:
                        text += '_self_say '
                    text += str(d['speech'][i + 1]) + '\n'
                    shown['speech'] = True
                used_current = True
            if (
                use_feat(opt, 'light_use_current_self_output', 'all')
                and label_type != 'action'
                and use_feat(opt, 'light_use_action', 'self')
                and d['action'][i + 1] is not None
            ):
                if 'remove' not in opt['light_use_current_self_output']:
                    text += '_self_act ' + str(d['action'][i + 1]) + '\n'
                    shown['action'] = True
                used_current = True
            if (
                use_feat(opt, 'light_use_current_self_output', 'all')
                and label_type != 'emote'
                and use_feat(opt, 'light_use_emote', 'self')
                and d['emote'][i + 1] is not None
            ):
                if 'remove' not in opt['light_use_current_self_output']:
                    text += '_self_emote ' + str(d['emote'][i + 1]) + '\n'
                    shown['emote'] = True
                used_current = True
            if (
                'all_filtered' in opt['light_use_current_self_output']
                and used_current is False
            ):
                label = None
            if label is not None:
                msg = {}
                msg['text'] = text
                msg['labels'] = label
                msg['model_name'] = model_name
                msg['did_continue'] = did_continue
                msg['score'] = score
                add_negs(
                    msg,
                    d,
                    i + 1,
                    label_type,
                    split,
                    int(opt.get('light_use_cands', 100)),
                    opt.get('light_use_affordances', True),
                )
                msgs.append(msg)
                text = ''
            if (
                use_feat(opt, 'light_use_speech', 'self')
                and d['speech'][i + 1] is not None
                and ('speech' not in shown)
            ):
                if opt['light_use_speech_prefix']:
                    text += '_self_say '
                elif opt['light_use_person_names_prefix']:
                    text += f"*{d['self_agent']['name']}*: "
                text += str(d['speech'][i + 1]) + '\n'
            if (
                use_feat(opt, 'light_use_action', 'self')
                and d['action'][i + 1] is not None
                and ('action' not in shown)
            ):
                text += '_self_act ' + str(d['action'][i + 1]) + '\n'
            if (
                use_feat(opt, 'light_use_emote', 'self')
                and d['emote'][i + 1] is not None
                and ('emote' not in shown)
            ):
                text += '_self_emote ' + str(d['emote'][i + 1]) + '\n'
    if len(msgs) > 0:
        msgs[-1]['episode_done'] = True
        for m in msgs:
            m['text'] = m['text'].rstrip('\n')
            # print(m.replace('\n', '\\n'))
            fix_labels(m, opt)
            global mx
            mx = max(len(m['label_candidates']), mx)
            # print(mx)
            txt = msg_to_str(m)
            fw.write(txt + '\n')


def write_alldata(opt, chats, dpath, ltype, split):
    # for now train, valid and test will all be identical
    fname = os.path.join(dpath, ltype + "_" + split + ".txt")
    fw_tst = io.open(fname, 'w')
    for chat in chats:
        chat = chat.copy()
        try:
            chat['self_agent'] = {
                'name': chat['human_persona']['name'],
                'persona': chat['human_persona']['persona'],
            }
            chat['partner_agent'] = {
                'name': chat['bot_persona']['name'],
                'persona': chat['bot_persona']['persona'],
            }
        except Exception as e:
            print(chat)
            raise (e)
        chat['setting'] = {
            'name': chat['location']['name'],
            'category': "Somewhere",  # TODO pull from DB?
            'description': chat['location']['description'],
        }
        turns = chat['dialogue']
        fin_turns = [t['text'] for t in turns]
        chat['speech'] = [None] + fin_turns
        chat['action'] = [None] * len(chat['speech'])
        chat['emote'] = [None] * len(chat['speech'])
        if opt['light_use_unseen_test']:
            setting = chat['location']['name'].lower()
            if setting not in unseen:
                #   import pdb; pdb.set_trace()
                write_dialog(opt, fw_tst, chat, ltype, split)
        else:
            write_dialog(opt, fw_tst, chat, ltype, split)
    fw_tst.close()


def add_negs(msg, d, ind, label_type, split, num_cands, use_affordances):
    if label_type == 'emote':
        msg['label_candidates'] = cands['emote']
    if label_type == 'which':
        msg['label_candidates'] = cands['which']
    if label_type == 'action':
        if use_affordances:
            msg['label_candidates'] = d['available_actions'][ind]
        else:
            msg['label_candidates'] = d['no_affordance_actions'][ind]
    if label_type == 'speech':
        cnt = 0
        label = msg['labels']
        negs = []
        # negs.append(label)
        used = {}
        # find 99 random negs that are != gold label
        while True:
            ind = rand.randint(0, len(cands['speech']) - 1)
            txt = cands['speech'][ind]
            if txt != label and ind not in used:
                negs.append(txt)
                used[ind] = True
                cnt += 1
                if cnt == num_cands - 1:
                    break
        # insert label in a random position
        negs.insert(rand.randrange(len(negs) + 1), label)
        msg['label_candidates'] = negs


cands = {}  # candidates


def write_out_candidates(chats, dpath, dtype):
    cands['speech'] = []
    for chat in chats:
        dialogue_turns = chat['dialogue']
        pos_cands = [t['text'] for t in dialogue_turns if t['id'] == 'human']
        cands['speech'] += pos_cands
    fw = io.open(os.path.join(dpath, "speech_" + dtype + "_cands.txt"), 'w')
    for k in cands['speech']:
        fw.write(k + "\n")
    fw.close


def read_dump(dump_path):
    with open(dump_path, 'r') as chat_file:
        chats = json.load(chat_file)

    if isinstance(chats, dict):
        headers = chats['header']
        rows = chats['rows']
        chats = []
        for row in rows:
            chat = {}
            for idx, header in enumerate(headers):
                chat[header] = row[idx]
            chats.append(chat)
    return chats


def remove_cand_path(in_path):
    fcp_split = in_path.split('fixed_candidates_path')
    pre_fixed_cands = fcp_split[0]
    post_fixed_cands = fcp_split[1].split('/')[-1]
    return pre_fixed_cands + "fixed_candidates_path=" + post_fixed_cands


def select_location_from_rough_string_match(location1, location2, chat):
    loc1_score = 0.0
    loc2_score = 0.0
    loc1_lower = location1.lower()
    loc2_lower = location2.lower()
    for turn in chat:
        for word in turn.lower().split(' '):
            if len(word) < 5:
                continue
            if word in loc1_lower:
                loc1_score += 1
            if word in loc2_lower:
                loc2_score += 1
    loc1_score /= len(loc1_lower.split(' '))
    loc2_score /= len(loc2_lower.split(' '))
    if loc1_score > loc2_score:
        return location1
    else:
        return location2


def format_chats_orig(chats):
    """
    Original hobbot chat format stored the dialogues as newline separated text, and the
    bot and human personas as strings.

    The choice and location appear in the dialogue history as strings.
    """
    chats = [
        c
        for c in chats
        if c['dialogue'] is not None and len(c['dialogue'].split('\\n')) > 1
    ]

    converted_chats = []
    for c in chats:
        try:
            # set up personas
            old_human_persona = c['human_persona'].split('\\n')
            c['human_persona'] = {
                'id': None,
                'name': old_human_persona[0],
                'persona': old_human_persona[1],
                'db_id': None,
            }
            old_bot_persona = c['bot_persona'].split('\\n')
            c['bot_persona'] = {
                'id': None,
                'name': old_bot_persona[0],
                'persona': old_bot_persona[1],
                'db_id': None,
            }

            old_dialogue_turns = c['dialogue'].split('\\n')
            if old_dialogue_turns[0].startswith("Human:"):
                # The very first 125 chats we collected didn't add the
                # specific location, so there are two possibilities.
                # We select the possible location with the
                # most word overlap with the chat.
                use_location = select_location_from_rough_string_match(
                    old_human_persona[2], old_bot_persona[2], old_dialogue_turns
                )
                old_dialogue_turns = [use_location] + old_dialogue_turns
            # extract location:
            location_turn_split = old_dialogue_turns[0].split(', ')
            c['location'] = {
                'id': None,
                'name': location_turn_split[0],
                'description': ', '.join(location_turn_split[1:]),
                'db_id': None,
            }

            # parse through dialogue
            new_dialogue = []
            c['choice'] = None
            BOT_TAG = 'Bot: '
            CHOICE_TAG = 'CHOICE: '
            HUMAN_TAG = 'Human: '
            c['score'] = int(c['score'])
            score_per_turn = c['score'] / max(1, len(old_dialogue_turns) / 2)
            for turn in old_dialogue_turns[1:]:
                if turn.startswith(BOT_TAG):
                    new_dialogue.append(
                        {'id': 'bot', 'type': 'speech', 'text': turn[len(BOT_TAG) :]}
                    )
                elif turn.startswith(HUMAN_TAG):
                    new_dialogue.append(
                        {
                            'id': 'human',
                            'type': 'speech',
                            'text': turn[len(HUMAN_TAG) :],
                            'score': score_per_turn,
                        }
                    )
                elif turn.startswith(CHOICE_TAG):
                    c['choice'] = {
                        'id': 'human',
                        'type': 'choice',
                        'text': turn[len(CHOICE_TAG) :],
                    }
            c['dialogue'] = new_dialogue
            c['is_complete'] = c['choice'] is not None
            c['did_continue'] = (
                c['choice'] is not None and c['choice']['text'] != 'EXIT'
            )
            converted_chats.append(c)
        except Exception:
            if c['ds'] == '2020-06-23':
                # We expect some chats from this day to have
                # broken formatting
                continue
            else:
                raise

    return converted_chats


def format_chats_june_2020(chats):
    """
    June 2020 format involves storing location and choice in the dialogue, but complete
    human and bot personas as objects that need to be json loaded.

    Model name also contains the full fixed candidates path, when we're only truly
    interested in the filename.
    """
    for c in chats:
        c['dialogue'] = json.loads(c['dialogue'].replace('\\\\', '\\'))
        c['human_persona'] = json.loads(c['human_persona'].replace('\\\\"', '\\"'))
        c['bot_persona'] = json.loads(c['bot_persona'].replace('\\\\"', '\\"'))
        c['model_name'] = remove_cand_path(c['model_name'].replace('\\\\"', '\\"'))
    chats = [c for c in chats if c['dialogue'] is not None and len(c['dialogue']) > 1]

    # Parse the choice and location out from the dialogue
    for c in chats:
        last_chat = c['dialogue'][-1]
        if last_chat['type'] == 'choice':
            c['choice'] = last_chat
            c['dialogue'] = c['dialogue'][:-1]
        else:
            c['choice'] = None

        c['location'] = c['dialogue'][0]
        c['dialogue'] = c['dialogue'][1:]
        c['score'] = int(c['score'])
        c['is_complete'] = c['choice'] is not None
        c['did_continue'] = c['choice'] is not None and c['choice']['text'] != 'EXIT'

        for t in c['dialogue']:
            if len(t['text'].strip()) == 0:
                t['text'] = '__SILENCE__'

    return chats


def format_chats(chat_list):
    if chat_list[0]['ds'] < '2020-06-24':
        return format_chats_orig(chat_list)
    else:
        return format_chats_june_2020(chat_list)


def read_dumps(dump_dir):
    dump_files = os.listdir(dump_dir)
    dump_files.sort()
    all_chats = []
    for dump_fn in dump_files:
        if not dump_fn.endswith('.json'):
            continue
        read_chats = read_dump(os.path.join(dump_dir, dump_fn))
        all_chats += format_chats(read_chats)
    return all_chats


def filter_chats(chats, return_flagged=False):
    chats = [c for c in chats if c['dialogue'] is not None and len(c['dialogue']) > 0]
    return [c for c in chats if (c['flagged_messages'] is not None) == return_flagged]


def remove_under_score_cutoff(chats, score_cutoff):
    return [c for c in chats if c['score'] >= score_cutoff]


def remove_over_score_cutoff(chats, score_cutoff):
    return [c for c in chats if c['score'] <= score_cutoff]


def remove_except_score_cutoff(chats, score_cutoff):
    return [c for c in chats if c['score'] == score_cutoff]


def filter_by_model(chats, model_name):
    return [c for c in chats if c['model_name'] == model_name]


def filter_by_chat_date(chats, chat_date):
    return [c for c in chats if c['ds'] <= chat_date]


def filter_by_continue_type(chats, continue_type):
    desired_continue_status = continue_type == 'continue'
    return [c for c in chats if c['did_continue'] == desired_continue_status]


def build_from_dump(opt, write_path, dump_dir):
    # set up chats
    chats = read_dumps(dump_dir)
    filtered_chats = filter_chats(chats)
    valid_choice_chats = filtered_chats[:10000]
    remaining_chats = filtered_chats[10000:]
    rand = random.Random(3)
    rand.shuffle(valid_choice_chats)

    # Construct a validation and test set with a preset cutoff
    above_test_cutoff = remove_under_score_cutoff(
        valid_choice_chats, score_cutoff=LIGHT_TEST_CUTOFF
    )
    below_test_cutoff = [
        c for c in valid_choice_chats if c['score'] < LIGHT_TEST_CUTOFF
    ]

    test_chats = above_test_cutoff[:1000]
    valid_chats = above_test_cutoff[1000:1500]
    remaining_chats += above_test_cutoff[1500:]

    trainable_chats = remaining_chats + below_test_cutoff
    if opt['light_use_hard_score_cutoff']:
        train_chats = remove_except_score_cutoff(
            trainable_chats, score_cutoff=opt['light_use_score_cutoff']
        )
    elif opt['light_use_max_score_cutoff'] > 0:
        train_chats = remove_over_score_cutoff(
            trainable_chats, score_cutoff=opt['light_use_max_score_cutoff']
        )
    else:
        train_chats = remove_under_score_cutoff(
            trainable_chats, score_cutoff=opt['light_use_score_cutoff']
        )

    if opt.get('light_model_name'):
        model_name = opt['light_model_name'].split('+')
        assert all(m in CURRENT_HOBBOT_MODELS for m in model_name)
        train_chats = sum([filter_by_model(train_chats, m) for m in model_name], [])

    if opt['light_use_continue_type'] != 'all':
        train_chats = filter_by_continue_type(
            train_chats, opt['light_use_continue_type']
        )

    if opt.get('light_use_date_cutoff'):
        date_cutoff = opt['light_use_date_cutoff']
        train_chats = filter_by_chat_date(train_chats, date_cutoff)

    write_out_candidates(test_chats, write_path, 'test')
    write_alldata(opt, test_chats, write_path, 'speech', 'test')

    write_out_candidates(valid_chats, write_path, 'valid')
    write_alldata(opt, valid_chats, write_path, 'speech', 'valid')

    write_out_candidates(train_chats, write_path, 'train')
    write_alldata(opt, train_chats, write_path, 'speech', 'train')
