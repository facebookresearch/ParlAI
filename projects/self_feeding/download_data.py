import parlai.core.params as params
from parlai.tasks.self_feeding.build import build


if __name__ == '__main__':
    opt = params.ParlaiParser().parse_args(print_args=False)
    build(opt)