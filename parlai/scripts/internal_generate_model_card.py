#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Script to generate the model card automatically.

(internal version)
"""
# FIXME: change if moved to public
from matplotlib.pyplot import get
from nltk import parse
import parlai_internal.scripts.label_subgroup_saver as lsav
from parlai.scripts.generate_model_card import *


def make_exs_tables(f_contents):
    """
    creates a table within a table for the number of examples in each subgroup
    `f_contents` should be a dictionary that contains the quantative analysis results of
    this format: {(datatype, subgroup): file_content}

    Sample table could look like this:
    |gender | gender2|
    :--:|:--:
    |<table>
    <tr><th> Datatype </th><th> all </th><th>female</th><th>gender-neutral</th><th>male</th></tr>
    <tr><td>test</td><td>6000</td><td>1459</td><td>2907</td><td>1634</tr></table>
    |<table>
    <tr><th> Datatype </th><th> all </th><th>female</th><th>gender-neutral</th><th>male </th></tr>
    <tr><td>test</td><td>6000</td><td>124</td><td>5277</td><td>599</tr></table>|
    """
    # subgroups keys are the title of the first level of table
    table_titles = list({subgroup for _, subgroup in f_contents})
    table1_header = f"|{' | '.join(table_titles)}|\n"
    table1_header += '|'.join([':--:'] * len(table_titles)) + '\n'

    # creating table
    table = ''
    subgroups_keys = {subgroup: [] for subgroup in table_titles}
    for dt, subgroup in f_contents:
        subgroups_keys[subgroup].append(dt)
    for subgroup_key in table_titles:
        table += '|<table>'
        #  first get subgroups as level 2 table headers
        exs_metric_keys = set()
        for i, dt in enumerate(subgroups_keys[subgroup_key]):
            report = f_contents[(dt, subgroup_key)]['report']
            exs_metric_keys = {metric for metric in report if 'exs' in metric}
            exs_metric_keys = sorted(list(exs_metric_keys))
            # headers
            if i == 0:
                header2 = []
                for key in exs_metric_keys:
                    header2.append('all')
                    if key != 'exs':
                        key = key.replace('.json', '')
                        header2[-1] = key.split('/')[-2].split('_')[-1]
                table += "<tr><th>Datatype</th><th>"
                table += f"{'</th><th>'.join(header2)} </th></tr>"
            # actual row
            row_content = [str(report[mkey]) for mkey in exs_metric_keys]
            table += f"<tr><td>{dt}</td><td>{'</td><td>'.join(row_content)}</tr>"
        table += '</table>'
    return table1_header + table + '|'


def regroup_datasets(opt, dt_task, label_qargs, special):

    base_folder = os.path.join(opt['folder_to_save'], 'data')
    os.makedirs(base_folder, exist_ok=True)

    # saving the arg info in the meta file
    meta_fname = os.path.join(opt['folder_to_save'], 'meta_files', 'label_qargs.json')
    os.makedirs(os.path.join(opt['folder_to_save'], 'meta_files'), exist_ok=True)
    with open(meta_fname, 'w+') as f:
        json.dump({'quant': label_qargs, 'spec': special}, f)

    err_mgs = []
    ignore_keys = {'folder_to_save', 'model', 'model_file', 'mode', 'evaltask'}
    parser = {k: v for k, v in opt['override'].items() if k not in ignore_keys}
    base_args = {key: val for key, val in label_qargs.items() if key != 'datatype'}
    parser.update(base_args)
    # reset other constants
    parser['save_by_subgroup'] = True
    parser['save_ext'] = '.json'

    for uniq in special:
        parser['model'] = uniq['model']
        parser['model_file'] = uniq['model_file']
        parser['data_parallel'] = False
        for dt in label_qargs['datatype']:
            os.makedirs(os.path.join(base_folder, uniq['subgroup'], dt), exist_ok=True)
            parser['save_loc'] = os.path.join(base_folder, uniq['subgroup'], dt, '_')
            # auto changes it so that lsav doesn't raise an error
            actual_dt = 'train:evalmode' if 'train' in dt else dt
            parser['datatype'] = actual_dt
            parser['task'] = ','.join(dt_task[dt])
            if opt['verbose']:
                extra_special_print(
                    f"working on {uniq['subgroup']} using model file {uniq['model_file']} on datatype {actual_dt}"
                )
                Opt(parser).log()

            try:
                lsav.LabelSubgroupSaver.main(**parser)
            except Exception:
                e = traceback.format_exc()
                msg = get_report_msg("label_subgroups", parser['save_loc'], e)
                extra_special_print(msg, color='red')
                err_mgs.append(msg)
    return err_mgs


def save_quant_eval(opt, datatypes, eval_qargs):
    # first get subgroups
    check = os.path.join(opt['folder_to_save'], 'data')
    # ignore anything with `.`; ie. files or hidden dir/files
    subgroups = [direct for direct in os.listdir(check) if '.' not in direct]

    # make sure folder for saving evaluation results exist
    save_folder = os.path.join(opt['folder_to_save'], 'quant_eval')
    os.makedirs(save_folder, exist_ok=True)

    err_msgs = []
    parser = eval_model.setup_args().parse_args([])
    parser = get_new_parser(parser, opt['override'])
    parser['aggregate_micro'] = True
    for subgroup in subgroups:
        check = os.path.join(opt['folder_to_save'], 'data', subgroup)
        for dt in datatypes:
            check2 = os.path.join(check, dt)
            files = [
                os.path.join(check2, file)
                for file in os.listdir(check2)
                if file.endswith('.json')
            ]
            tasks = [f'jsonfile:jsonfile_datapath={file}' for file in files]
            tasks = ','.join(tasks)
            save_fname = os.path.join(save_folder, f'quant_eval_{subgroup}_{dt}.json')
            # command = f"parlai em -mf {opt['model_file']} -t {tasks} -dt {dt} -bs {opt['batchsize']} --aggregate_micro True -rf {save_fname} {' '.join(base_args)}"
            # extra_special_print(command)
            parser['report_filename'] = save_fname
            parser['datatype'] = dt
            parser['tasks'] = tasks

            try:
                _ = eval_model.eval_model(parser)
                # _ = eval_model.EvalModel.main(
                #     # model_file=opt['model_file'],
                #     # task=tasks,
                #     # aggregate_micro=True,
                #     # batchsize=opt['batchsize'],
                #     # report_filename=save_fname,
                #     # datatype=dt,
                #     **eval_qargs,
                # )
            except Exception:
                e = traceback.format_exc()
                msg = get_report_msg('eval for quant analysis', save_fname, e)
                extra_special_print(msg, color='red')
                err_msgs.append(msg)
    return err_msgs


# set containing internal tasks where we can just remove "internal:" and it will be okay
internal_remove_tasks = {
    "internal:blended_skill_talk",
    "internal:light_dialog",
    "internal:light_dialog",
    "internal:eli5",
    "internal:igc",
    "internal:wizard_of_internet",
}

# dictionary mapping original task to new location, if they exist
remap_task = {
    "internal:safety:wikiToxicComments": "dialogue_safety:wikiToxicComments",
    "internal:safety:adversarial": "dialogue_safety:adversarial",
    "internal:personal_knowledge:PersonalTopicFollowup": "msc:PersonaSummary",
    # TODO: check if this is actually correct
    "internal:safety:multiturnConvAI2": "dialogue_safety:multiturn",
    "internal:safety:boring": None,
    "internal:convai2_review": None,
    "internal:safety:boringConvAI2Review": None,
    "internal:comment_battle:ImageDialogGenerationTeacher": None,
    "internal:safety:adversarialConvAI2Review": None,
    "internal:new_reddit:small": None,
}


@register_script('internal_generate_model_card', aliases=['igmc'])
class InternalGenerateModelCard(GenerateModelCard):
    def process_task(self, task, ignore_task=True):
        """
        tries to remap tasks to their external version, and then may ignore the tasks w/o
        ext.

        version depending on `ignore_task`
        """
        # processing tasks so that no arguments are included
        # unless it's a fromfile or jsonfile one
        splitted = task.split(':')
        stop = len(splitted)
        if 'fromfile:' not in task and 'jsonfile:' not in task:
            for i in range(len(splitted)):
                if '=' in splitted[i]:
                    stop = i
                    break
        actual_task = ':'.join(splitted[:stop])

        # using actual task, figure out if it should be redirected
        if actual_task in internal_remove_tasks:
            return task.replace('internal:', '')
        if actual_task in remap_task:
            if remap_task.get(actual_task):
                return remap_task[actual_task] + ':'.join(splitted[stop:])
            else:
                return None
        if 'fromfile:' in task or 'jsonfile:' in task or 'internal:' in task:
            return None if ignore_task else task
        return task

    ##########################################
    # generation mode class functions
    ##########################################
    def save_quant_analysis(self, qargs):
        tt, et = (self.train_tasks, self.eval_tasks)
        dt_task = {'train': tt, 'test': et, 'valid': et}
        err_msgs = []
        label = qargs['label_qargs']
        errs = regroup_datasets(self.opt, dt_task, label, qargs['model_qargs'])
        err_msgs.extend(errs)
        errs2 = save_quant_eval(self.opt, label['datatype'], qargs['eval_qargs'])
        err_msgs.extend(errs2)
        return errs

    def get_quant_args(self):
        """
        Processes quantitative arugments used for label saving and evaluation. Uses get_args and then checks to make sure that eval_qargs always has the three
        arguments: model, model_file, and subgroup.

        If no `defaults` are passed in, it uses `zoo:md_gender/model` as the model_file,
        `parlai_internal.projects.model_cards_subgroup.agents:SubgroupBertRankerClassifier`
        as the model, gender as its subgroup name, and only the test as its datatype.
        """
        # deal with defaults and then call get_args
        qargs_default = {
            'label_qargs': {'datatype': ['test']},
            'model_qargs': [
                {
                    # FIXME: when moving to public
                    'model': 'parlai_internal.projects.model_cards_subgroup.agents:SubgroupBertRankerClassifier',
                    'model_file': 'zoo:md_gender/model',
                    'subgroup': 'gender',
                }
            ],
            'eval_qargs': {},
        }
        qargs = self.get_args(qargs_default)

        # checking to make sure that special args always has them
        sorted_expected = ['model', 'model_file', 'subgroup']
        for arg in qargs_default['model_qargs']:
            if len(arg) != 3 or sorted(arg.keys()) != sorted_expected:
                err_msg = f"model_args items should always have the following 3 keys: {', '.join(sorted_expected)}. Currently, it has these keys: {', '.join(arg.keys())}"
                raise RuntimeError(err_msg)
        return qargs

    def _gen_jobs(self):
        jobs, args = super()._gen_jobs()
        if self.model_type == CLASSIFIER:
            jobs.add('quant_analysis')
            args['quant_analysis_args'] = self.get_quant_args()
        return jobs, args

    ##########################################
    # editing/final mode class functions
    ##########################################

    def _setup_quant(self):
        """
        This does setup for quantitative analysis:

        - gets the metrics involved and adds the validation metric while
            removing `exs` & removes underscores for names
        - reads in the quantitative reports
        - filters so that we only end up with ones specified by user
            (using -qargs)
        returns the file contents, metric names, metrics, and datatypes used
        """
        # metrics: add validation metrics and remove 'exs'
        metrics = set(self.opt.get('quant_metrics', []))
        metrics.add(self.model_opt.get('validation_metric', 'exs'))
        metrics.discard('exs')
        print('metrics', metrics)
        mnames = [re.sub(r'_+', ' ', metric) for metric in metrics]

        # determine which files to read
        files = self.opt.get('quantitative_report_files')
        if len(files) == 0:
            # search for the files in the current directory
            files = [
                os.path.join(self.opt['folder_to_save'], 'quant_eval', file)
                for file in os.listdir(f"{self.opt['folder_to_save']}/quant_eval")
                if file.endswith('.json')
            ]
        # get all file contents or just the specified ones from -qargs
        f_contents, general = ({}, {})
        general['subgroup'] = set(self.opt.get('subgroup', []))
        dt = self.opt.get('quant_datatype', '').split(',')
        general['datatype'] = set(filter(None, dt))
        add_dt = len(general['datatype']) == 0
        add_subgroup = len(general['subgroup']) == 0
        for file in files:
            splitted = file.replace('.json', '').split('/')[-1].split('_')
            subgroup, dt = ('_'.join(splitted[2:-1]), splitted[-1])
            if add_dt:
                general['datatype'].add(dt)
            if add_subgroup:
                general['subgroup'].add(subgroup)
            if dt in general['datatype'] and subgroup in general['subgroup']:
                with open(file, 'rb') as f:
                    f_contents[(dt, subgroup)] = json.load(f)
        return f_contents, mnames, metrics, general['datatype']

    def quantitative_analyses(self):
        # getting required info and setting up initial variables
        f_contents, mnames, metrics, dts = self._setup_quant()
        models, fts = ([], self.opt['folder_to_save'])
        meta_fname = os.path.join(fts, 'meta_files', 'label_qargs.json')
        if os.path.isfile(meta_fname):
            # reading in previous args (separated into general and special)
            with open(meta_fname, 'rb') as f:
                _, spec = json.load(f).values()
            for model_l in spec:
                # get the model dictionary
                zpath = to_zoo(self.opt, model_l['model_file'])
                mdict = all_models.get(zpath, {'title': zpath})
                # get the link and description if there exists one
                default = "https://parl.ai/docs/zoo.html/"
                default += f"{'-'.join(mdict['title'].split())}"
                models.append(make_link(mdict['title'], mdict.get('project', default)))
                models[-1] += f" ({model_l['subgroup']})"
                if mdict.get('description'):
                    models[-1] += ': ' + mdict.get('description')
                models[-1] += '\n'
            msg = f"The datasets used were re-labeled into different subgroups by the following classifier(s):\n - {'- '.join(models)}\n\n Note that each datatype contains all of the tasks or datasets; ie. for the test datatype, it contains all the evaulation datsets. **Since the data was labeled by a classifier, the results below should only be used as an approximate for how well this model will do in real-world situations.**"
        else:
            # if meta_files/label_qargs.json doesn't exist,
            # then we don't really know where the quant file came from
            msg = ''
            if self.mode == self.EDIT:
                msg = create_warning('Add Quantitative Analysis Message!!!!!')
        content = ['## Quantitative Analysis', msg]

        # add table w/ number of examples
        content.append("Here's a table of the number of examples within each subgroup:")
        content.append(make_exs_tables(f_contents))

        # heatmap settings
        import seaborn as sns
        import pandas as pd

        args = {'cmap': sns.color_palette("Greens", as_cmap=True)}
        images = {dt: [] for dt in dts}

        for dt, subgroup in f_contents:
            report = f_contents[(dt, subgroup)]['report']
            # create the dataframe for heatmaps : {by subgroup, all}
            stats_dict = [
                {metric: {} for metric in mnames},
                {
                    re.sub(r'_+', ' ', key): {'all': report[key]}
                    for key in report
                    if key in metrics
                },
            ]
            for key in report:
                splitted = key.split('/')
                metric = re.sub(r'_+', ' ', splitted[-1])
                if metric in mnames and len(splitted) > 1:
                    _subgroup = splitted[-2].split('.')[0].split('_')[-1]
                    stats_dict[0][metric][_subgroup] = report[key]
            stats = [pd.DataFrame(d).sort_index(axis=1) for d in stats_dict]
            # create heatmaps
            fname = os.path.join(fts, f'quant_graph_{dt}_{subgroup}.png')
            _, _ = get_heatmap(stats, fout=fname, heatmapkws_user=args, title=subgroup)
            images[dt].append(f'quant_graph_{dt}_{subgroup}.png')
        # section images by datatype
        for dt, imgs in images.items():
            content.append(f'### {dt.capitalize()} Set')
            imgs_l = make_img_links(imgs, width='600')
            content.append(f"<p>{imgs_l} </p>")
        return '\n\n'.join(content)

    def extra_analysis(self):
        if self.model_type == GENERATOR:
            return self.safety_benchmark()
        elif self.model_type == CLASSIFIER:
            return self.quantitative_analyses()


if __name__ == '__main__':
    InternalGenerateModelCard.main()
