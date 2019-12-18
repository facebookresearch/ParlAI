#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import io
import os
import pickle
from parlai.utils.misc import msg_to_str

rand = random.Random(42)


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
    if opt['light_use_objects']:
        for o, od in d['all_descriptions'].items():
            text += '_object_desc ' + o + " : " + od + '\n'
        if False:
            for o in d['room_objects'][0]:
                text += '_object_in_room ' + o + '\n'
            for o in d['carrying'][0]:
                text += '_object_carrying ' + o + '\n'
            for o in d['wearing'][0]:
                text += '_object_wearing ' + o + '\n'
            for o in d['wielding'][0]:
                text += '_object_wielding ' + o + '\n'
    for i in range(0, l, 2):
        if i < l - 1:
            if (
                use_feat(opt, 'light_use_speech', 'partner')
                and d['speech'][i] is not None
            ):
                if opt['light_use_speech_prefix']:
                    text += '_partner_say '
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
            used_current = False
            shown = {}
            if (
                use_feat(opt, 'light_use_current_self_output', 'all')
                and label_type != 'speech'
                and use_feat(opt, 'light_use_speech', 'self')
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
            # print(m.replace('\n', '\\n'))
            fix_labels(m, opt)
            global mx
            mx = max(len(m['label_candidates']), mx)
            # print(mx)
            txt = msg_to_str(m)
            fw.write(txt + '\n')


def write_alldata(opt, db, dpath, ltype, split):
    # for now train, valid and test will all be identical
    fname = os.path.join(dpath, ltype + "_" + split + ".txt")
    fw_tst = io.open(fname, 'w')
    for d in db:
        if d['split'] != split:
            continue
        d = d.copy()
        d['self_agent'] = d['agents'][1]
        d['partner_agent'] = d['agents'][0]
        write_dialog(opt, fw_tst, d, ltype, split)

        # now flip the conversation around so the target is the other speaker..
        d2 = d.copy()
        d2['self_agent'] = d2['agents'][0]
        d2['partner_agent'] = d2['agents'][1]
        d2['speech'] = list(d2['speech'])
        d2['speech'].insert(0, None)
        d2['emote'] = list(d2['emote'])
        d2['emote'].insert(0, None)
        d2['action'] = list(d2['action'])
        d2['action'].insert(0, None)
        d2['available_actions'] = list(d2['available_actions'])
        d2['available_actions'].insert(0, None)
        d2['no_affordance_actions'] = list(d2['no_affordance_actions'])
        d2['no_affordance_actions'].insert(0, None)
        write_dialog(opt, fw_tst, d2, ltype, split)
    fw_tst.close()


def add_negs(msg, d, ind, label_type, split, num_cands, use_affordances):
    if label_type == 'emote':
        msg['label_candidates'] = cands['emote']
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


def write_out_candidates(db, dpath, dtype):
    emotes = {}
    acts = {}
    cands['emote'] = []
    cands['action'] = []
    cands['speech'] = []
    for d in db:
        gen_no_affordance_actions_for_dialogue(d)
        if d['split'] != dtype:
            continue
        for e in d['emote']:
            if e is not None and e not in emotes:
                cands['emote'].append(e)
                emotes[e] = True
        for act in d['action']:
            if act is not None and act not in acts:
                cands['action'].append(act)
                acts[act] = True
        for s in d['speech']:
            if s is not None:
                cands['speech'].append(s)
    for l in ['emote', 'speech', 'action']:
        fw = io.open(os.path.join(dpath, l + "_" + dtype + "_cands.txt"), 'w')
        for k in cands[l]:
            fw.write(k + "\n")
        fw.close


def build_from_db(opt, db_path, data_path, fname, fname2):
    # set up database
    dbp = os.path.join(db_path, fname)
    file = open(dbp, 'rb')
    db = pickle.load(file)
    # set up unseen database
    dbp = os.path.join(db_path, fname2)
    file = open(dbp, 'rb')
    db_unseen = pickle.load(file)

    # add test set labeling
    for i in range(0, len(db)):
        db[i]['split'] = 'train'
    for i in range(0, len(db_unseen)):
        db_unseen[i]['split'] = 'test_unseen'
    rand2 = random.Random(42)
    x = []
    # choose 1000 dialogues from the set that contain object manipulations
    # (hard-coded here).
    for i in range(1368, 4733):
        x.append(i)
    rand2.shuffle(x)
    for i in range(0, 1000):
        db[x[i]]['split'] = 'test'
    for i in range(1000, 1500):
        db[x[i]]['split'] = 'valid'

    for split in ['train', 'valid', 'test']:
        write_out_candidates(db, data_path, split)
        write_alldata(opt, db, data_path, 'speech', split)
        write_alldata(opt, db, data_path, 'action', split)
        write_alldata(opt, db, data_path, 'emote', split)

    for split in ['test_unseen']:
        write_out_candidates(db_unseen, data_path, split)
        write_alldata(opt, db_unseen, data_path, 'speech', split)
        write_alldata(opt, db_unseen, data_path, 'action', split)
        write_alldata(opt, db_unseen, data_path, 'emote', split)
