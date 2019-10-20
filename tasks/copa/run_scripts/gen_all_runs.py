from itertools import product
import os
import sys
from os import path


def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def main(script_dir):
    sent_reprs = ['avg', 'max', 'diff', 'diff_sum', 'coherent', 'attn', 'coref']

    all_combs = []
    all_combs += list(product(["bert", "spanbert", "xlnet", "roberta"],
                              ["base", "large"], sent_reprs))
    all_combs += list(product(["gpt2"],  ["small", "medium", "large"], sent_reprs))

    script_path = "/home/shtoshni/Research/causality/tasks/copa/main.py"
    with open(path.join(script_dir, "run.sh"), 'w') as f:
        for comb in all_combs:
            arg_str = (" -model " + comb[0] + " -model_size " + comb[1]
                       + " -pool_method " + comb[2])
            f.write("python " + script_path + arg_str + "\n")


if __name__ == '__main__':
    script_dir = get_script_path()
    main(script_dir)
