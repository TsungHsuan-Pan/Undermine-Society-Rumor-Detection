from utils import (
    get_global_csv_path,
    set_global_fp
)
import os
import csv
from run import (
    setup_args,
    special_rules,
    set_output_dir_name,
    add_language_in_train_dev_test_files
)


args = setup_args()
special_rules(args)
set_output_dir_name(args)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
add_language_in_train_dev_test_files(args)


def metadata_to_global(f_global):
    row = ["model_name", "accuracy", 
        "weighted_precision", "weighted_recall", "weighted_f1"]
    write = csv.writer(f_global)
    write.writerow(row)

global_csv_path = get_global_csv_path(args.taxonomy)
if os.path.exists(global_csv_path):
    os.remove(global_csv_path)
f_global = set_global_fp(args)
metadata_to_global(f_global)

