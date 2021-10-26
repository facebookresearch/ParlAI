import json
from argparse import ArgumentParser
from pprint import pprint
import textwrap

wrapper = textwrap.TextWrapper(initial_indent="\t", width=50, subsequent_indent="\t")

no_indent_wrapper = textwrap.TextWrapper(
    initial_indent="", width=50, subsequent_indent=""
)

parser = ArgumentParser()

parser.add_argument("-fn", "--filename", type=str, help="world_logs file")

args = parser.parse_args()


def format_context(context):

    split1 = context.split("<user>")
    instruction = split1[0]
    f_context = f"Instruction: {instruction}\n"
    print(no_indent_wrapper.fill(f_context))
    for s in split1[1:]:
        split2 = s.split("<system>")
        if len(split2) == 2:
            user_utt, system_utt = split2
            print(no_indent_wrapper.fill(user_utt))
            print(wrapper.fill(system_utt))
        elif len(split2) == 1:
            user_utt = split2[0]
            print(no_indent_wrapper.fill(user_utt))


with open(args.filename, "r") as f:
    data = f.read().splitlines()

# for regular multiwoz output logs
# for d in data:
#     dict_ = json.loads(d)

#     context = dict_['dialog'][0][0]['text']

#     count = context.count("<user>")

#     label = dict_['dialog'][0][0]['eval_labels'][0]
#     metrics = dict_['dialog'][0][1]['metrics']
#     jga = metrics.get('joint goal acc', -1)
#     pred = dict_['dialog'][0][1]['text']
#     if jga == 0.0 and count ==1:
#         formatted_convo= format_context(context)
#         print(f"\n\tpred: {pred}\n\tgold: {label}")
#     # if dict_[""]
#         break


# for LAUG logs
for idx in range(0, len(data), 2):
    d = data[idx]
    d_inv = data[idx + 1]
    dict1 = json.loads(d)
    dict2 = json.loads(d_inv)

    context1 = dict1['dialog'][0][0]['text']
    context2 = dict2['dialog'][0][0]['text']

    count = context1.count("<user>")

    label = dict1['dialog'][0][0]['eval_labels'][0]
    metrics = dict1['dialog'][0][1]['metrics']
    jga = metrics.get('jga_original', -1)
    pred = dict1['dialog'][0][1]['text']

    label2 = dict2['dialog'][0][0]['eval_labels'][0]
    metrics2 = dict2['dialog'][0][1]['metrics']
    jga2 = metrics2.get('jga_perturbed', -1)
    pred2 = dict2['dialog'][0][1]['text']

    assert label == label2

    # print(jga, jga2)
    if jga == 0.0 and jga2 == 0.0 and count == 3:
        format_context(context1)
        print(f"\n\tpred: {pred}\n\tgold: {label}")

        format_context(context2)
        print(f"\n\tpred: {pred2}\n\tgold: {label2}")

        print("-" * 50 + "\n\n")

        # break
