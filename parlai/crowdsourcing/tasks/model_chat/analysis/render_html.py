#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import pandas as pd

from parlai.core.params import ParlaiParser
from IPython.core.display import HTML
from parlai_internal.projects.chats_render_utils.convo_collector import (
    is_bot_name,
    build_table,
    build_table_from_acute_format,
    build_table_from_interactive,
    build_table_from_qfunction,
    build_table_from_mephisto_run,
    BOT,
    HUMAN,
    DIALOGUE_OUTPUT_KEY,
    CONTEXT_OUTPUT_KEY,
    MODEL_NAME_KEY,
)
from parlai.utils.safety import OffensiveLanguageClassifier, OffensiveStringMatcher

SKIP_KEYS = ['turn_idx']


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, False, 'Render Logs in HTML format')
    parser.add_argument(
        '--html-path', type=str, default='', help='path storing rendered html files'
    )
    parser.add_argument(
        '--hide-speakername',
        type=bool,
        default=False,
        help='whether to hide the speaker name in the conversation',
    )
    parser.add_argument(
        '--logs-path', type=str, default=None, help='input path for build_table'
    )
    parser.add_argument(
        '--acute-logs-path',
        type=str,
        default=None,
        help='input path of build_table_from_acute_format',
    )
    parser.add_argument(
        '--interactive-logs-folder',
        type=str,
        default=None,
        help='input folder of build_table_from_interactive',
    )
    parser.add_argument(
        '--qfunction-csv-path',
        type=str,
        default=None,
        help='input path of build_table_from_qfunction',
    )
    parser.add_argument(
        '--mephisto-run-path',
        type=str,
        default=None,
        help='input path of build_table_from_mephisto_run',
    )
    parser.add_argument(
        '--preferred-bot-name',
        type=str,
        default=None,
        help='preferred name for the bot to be displayed',
    )
    parser.add_argument(
        '--cherry-pick',
        type=bool,
        default=False,
        help='whether to enable cherry-picking checkbox',
    )
    parser.add_argument(
        '--display-safety',
        type=bool,
        default=False,
        help='whether to enable cherry-picking checkbox',
    )
    parser.add_argument(
        '--custom-js-file',
        type=str,
        default=None,
        help='whether to include a custom JS file at the top of the HTML',
    )
    parser.add_argument(
        "--hhchat-datatype", type=str, choices=['train', 'valid', 'test']
    )
    parser.add_argument("--hhchat-session-id", type=int)
    parser.add_argument(
        "--annotation-datatype", type=str, choices=['train', 'valid', 'test']
    )
    parser.add_argument("--annotation-session-id", type=int)
    return parser


def is_convo_human_human(row):
    turn_cnt = 0
    bot_turn = 0
    for utterance in row[DIALOGUE_OUTPUT_KEY]:
        if utterance['id'] == 'context':
            continue
        turn_cnt += 1
        if is_bot_name(utterance['id']):
            bot_turn += 1
        if turn_cnt == 2:
            break
    return bot_turn != 1


def render_row(
    row_id, row, hide_name, preferred_bot_name, cherry_pick, safety_detector
):
    is_human_human = is_convo_human_human(row)
    result = []

    # Show context
    align = 'center'
    color = "green"
    bgcolor = '#31f723'
    for _, turn in enumerate(row[CONTEXT_OUTPUT_KEY]):
        if 'persona: ' in turn['text']:
            texts = turn['text'].split('\n')
            result.append(
                (
                    '<div style="white-space: pre-wrap; width:800px; margin-bottom: 30px; padding: 0.5em 1em;clear: both;  float: {}; color: {}; background-color: {}; border-radius: 5em">'
                    '{}'
                    '</div>'
                ).format(align, 'white', '#2391f7', texts[0])
            )
            result.append(
                (
                    '<div style="white-space: pre-wrap; width:800px; margin-bottom: 30px; padding: 0.5em 1em;clear: both;  float: {}; color: {}; background-color: {}; border-radius: 5em">'
                    '{}'
                    '</div>'
                ).format(align, 'black', '#e1e1e7', texts[1])
            )
            result.append(
                (
                    '<div style="white-space: pre-wrap; width:800px; margin-bottom: 30px; padding: 0.5em 1em;clear: both;  float: {}; color: {}; background-color: {}; border-radius: 5em">'
                    '{}'
                    '</div>'
                ).format(align, color, bgcolor, texts[2])
            )
        if turn.get('image_src') is not None:
            result.append(
                f'<div style="text-align: center; white-space: pre-wrap; width:800px; margin-bottom: 30px; padding: 0.5em 1em;clear: both;  float: {align}; color: {color}; background-color: {bgcolor}; border-radius: 5em"><img src={turn["image_src"]} alt="Image"/></div>'
            )
        if turn.get('problem_data') is not None:
            sentences = [f'{k}: {v}' for k, v in turn.get('problem_data').items()]
            text = '&#10&#10'.join(sentences)
            result.append(
                (
                    '<div style="white-space: pre-wrap; width:800px; margin-bottom: 30px; padding: 0.5em 1em;clear: both;  float: {}; color: {}; background-color: {}; border-radius: 5em">'
                    '{}'
                    '</div>'
                ).format(align, color, bgcolor, text)
            )

    # Show dialogue
    has_shown_botname = False
    chat_start_ts = -1
    for i, turn in enumerate(row[DIALOGUE_OUTPUT_KEY]):
        speakername = turn['id']
        text = turn['text']
        if is_human_human:
            is_bot = i % 2 == 1
            speakername = HUMAN
        else:
            is_bot = is_bot_name(speakername)
            if is_bot:
                if preferred_bot_name is not None:
                    speakername = preferred_bot_name
                elif has_shown_botname:
                    speakername = BOT
                has_shown_botname = True

        align = 'right' if is_bot else 'left'
        color = "white" if is_bot else "black"
        bgcolor = '#2391f7' if is_bot else '#e1e1e7'
        annotation_color = "white" if is_bot else "black"
        annotation_bgcolor = '#f06413' if is_bot else '#e1e42c'

        result.append(
            (
                '<div style="width:800px; margin:0 auto; overflow: auto; padding: 1ex 0;">'
                '<div style="clear: both; float: {}; color: {}; background-color: {}; padding: 0.5em 1em; border-radius: 5em; max-width: 80%">'
            ).format(align, color, bgcolor)
        )
        if hide_name:
            result.append(('<p style="margin: 0">{}</p>').format(text))
        else:
            result.append(
                ('<p style="margin: 0"><b>{}</b>: {}</p>').format(speakername, text)
            )

        # TODO: this is hard coded here; use you customized key
        if 'first_turn_start_ts' in turn:
            chat_start_ts = turn['first_turn_start_ts']

        annotation_result = []
        annotation_result.append(
            (
                '</div>'
                '<div style="clear: both; float: {}; color: {}; background-color: {}; padding: 0.5em 1em; border-radius: 1em; max-width: 80%">'
            ).format(align, annotation_color, annotation_bgcolor)
        )
        has_annotation = False

        # TODO: add additional flag to turn that off if it's not necessary
        if 'text' in turn and safety_detector is not None:
            for detector_name, detector in safety_detector.items():
                if turn['text'] in detector:
                    has_annotation = True
                    annotation_result.append(
                        (
                            '<div>'
                            '<input type= "checkbox" id= {} name= {} checked>'
                            '<label for={}>{}</label>'
                            '</div>'
                        ).format(
                            detector_name,
                            detector_name,
                            detector_name,
                            detector_name.capitalize() + ': not_ok',
                        )
                    )

        if 'problem_data' in turn:
            turn_problem_data = turn['problem_data']
            for k, v in turn_problem_data.items():
                if k in SKIP_KEYS:
                    continue
                if isinstance(v, str) and 'not_ok' in v or v is True:
                    if k == 'none_all_good':
                        continue
                    has_annotation = True
                    annotation_result.append(
                        (
                            '<div>'
                            '<input type= "checkbox" id= {} name= {} checked>'
                            '<label for={}>Turker label {}: {}</label>'
                            '</div>'
                        ).format(k, k, k, k, v)
                    )
                elif 'rating' in k:
                    has_annotation = True
                    annotation_result.append((f'<div> Rating = {v}</div>'))
                # TODO: hard coded here as key of the chat end timestamps
                elif k == 'chat_end_ts':
                    has_annotation = True
                    chat_duration_ts = (
                        turn_problem_data['chat_end_ts'] - chat_start_ts
                    ) / 60
                    annotation_result.append(
                        (f'<div> Chat Duration = {chat_duration_ts:.2f} min</div>')
                    )
                else:
                    # General case: just show the keys and values as-is
                    has_annotation = True
                    annotation_result.append(f'<div>{k}: {v}</div>')

        if has_annotation:
            result.extend(annotation_result)
        result.append(('</div></div>'))

    dialogue = (
        '<div style="background-color: white; margin: 0em; padding: 0.5em; '
        'font-family: sans-serif; font-size: 12pt; width: 60%;">'
        + ''.join(result)
        + '</div>'
    )
    checkbox = (
        '<div><input type= "checkbox" id= "cherry" name= "cherry">'
        '<label for="cherry">Cherry</label>'
        '</div>'
        '<div><input type= "checkbox" id= "lemon" name= "lemon">'
        '<label for= "lemon">Lemon</label>'
        '</div>'
        '<div><input type= "checkbox" id= "neutral" name= "neutral">'
        '<label for= "neutral">Neutral</label>'
        '</div>'
    )
    if cherry_pick:
        return HTML(
            f'<tr><td style="text-align: center">Pair {str(row_id)}</td><td>{checkbox}</td><td style="padding-left: 100px;">{dialogue}</td></tr>'
        )
    else:
        return HTML(
            f'<tr><td style="text-align: center">Pair {str(row_id)}</td style="text-align: center"><td style="padding-left: 100px;">{dialogue}</td></tr>'
        )


def render_many_conversations(
    table, hide_name, preferred_bot_name, cherry_pick, safety_detector
):
    table_rows = [
        render_row(
            idx, row, hide_name, preferred_bot_name, cherry_pick, safety_detector
        ).data
        for idx, (_, row) in enumerate(table.iterrows())
    ]
    if cherry_pick:
        body = HTML(
            f'<table border=1 frame=void rules=rows><tr><th>   </th><th>  Comments  </th><th style="padding-left:100px">Conversation</th></tr>{"".join(table_rows)}</table>'
        )
    else:
        body = HTML(
            f'<table border=1 frame=void rules=rows><tr><th>   </th><th style="padding-left:100px">Conversation</th></tr>{"".join(table_rows)}</table>'
        )
    result = f"<h2>{preferred_bot_name if preferred_bot_name is not None else ''}</h2><body>{body.data}</body>"
    return result


def render_many_conversations_by_model(
    table, hide_name, preferred_bot_name, cherry_pick, safety_detector
):
    model_list = list(table[MODEL_NAME_KEY].unique())
    result = '\
    <div id="toc_container">\
        <p class="toc_title">Model Pairs</p>\
            <ul class="toc_list">'
    for _, model_name in enumerate(model_list):
        result += '<li><a href="#{}">{}</a></li>'.format(model_name, model_name)
    result += '</ul></div>'
    for model_name in model_list:
        result += '<h2 id="{}"><li><a href="#toc_container">{}</a></li></h2><body>{}</body>'.format(
            model_name,
            model_name,
            render_many_conversations(
                table[table[MODEL_NAME_KEY] == model_name],
                hide_name,
                preferred_bot_name,
                cherry_pick,
                safety_detector,
            ),
        )

    return result


def render_html(table, args):
    """
    Save results to a certain path.
    """
    safety_detector = None
    if args['display_safety']:
        safety_clf = OffensiveLanguageClassifier()
        safety_sm = OffensiveStringMatcher()
        safety_detector = {'safety_clf': safety_clf, 'safety_sm': safety_sm}
    with open(args['html_path'], 'w+') as f:
        main_html = render_many_conversations_by_model(
            table,
            args['hide_speakername'],
            args['preferred_bot_name'],
            args['cherry_pick'],
            safety_detector,
        )
        if args['custom_js_file']:
            with open(args['custom_js_file'], 'r') as js_file:
                main_html = (
                    f'<script type="text/javascript">{js_file.read()}</script>'
                    + main_html
                )
        f.write(main_html)
    print(
        'To visualize conversations result, try scp username@devfair: {}'
        ' to your local machine'.format(args['html_path'])
    )


def render_live_mturk(json_file, output_html_path, display_safety=True):
    parser = setup_args()
    args = parser.parse_args()
    args['html_path'] = output_html_path
    args['display_safety'] = display_safety
    table = build_table(json_file)
    render_html(table, args)


def main():
    """
    python parlai_internal/projects/chats_render_utils/render_html.py --acute-logs-path
    parlai_internal/projects/chats_render_utils/samples_logs_format/acute_format_log.jso
    n --logs-path parlai_internal/projects/chats_render_utils/samples_logs_format/sample_logs.json.

    --interactive-logs-folder
    parlai_internal/projects/chats_render_utils/samples_logs_format --qfunction-csv-path
    parlai_internal/projects/chats_render_utils/samples_logs_format/sample_qfuction_chat
    .csv --html-path parlai_internal/projects/chats_render_utils/render_sample_html.html

    
    """
    parser = setup_args()
    args = parser.parse_args()
    if args['html_path'] == '' or args['html_path'] is None:
        raise ValueError('Output path for html is required.')
    table = pd.DataFrame()
    if args['logs_path'] is not None:
        table = table.append(build_table(args['logs_path']), ignore_index=True)
    if args['acute_logs_path'] is not None:
        table = table.append(
            build_table_from_acute_format(args['acute_logs_path']), ignore_index=True
        )
    if args['interactive_logs_folder'] is not None:
        table = table.append(
            build_table_from_interactive(args['interactive_logs_folder']),
            ignore_index=True,
        )
    if args['qfunction_csv_path'] is not None:
        all_conversations_df = pd.read_csv(args['qfunction_csv_path'], engine="python")
        table = table.append(
            build_table_from_qfunction(all_conversations_df), ignore_index=True
        )

    if args['mephisto_run_path'] is not None:
        table = table.append(
            build_table_from_mephisto_run(
                run_directory=args['mephisto_run_path'],
                onboarding_in_flight_data_file=os.path.join(
                    parser.parlai_home,
                    'parlai_internal/mturk/tasks/q_function/self_chat/json/onboarding_in_flight.jsonl',
                ),
            ),
            ignore_index=True,
        )

    render_html(table, args)


if __name__ == "__main__":
    main()
