import os
from run import (
    setup_args,
    special_rules,
    set_output_dir_name,
    add_language_in_train_dev_test_files
)
from utils import (
    write_label_metadata_to_file,
)
import shutil

args = setup_args()
special_rules(args)
set_output_dir_name(args)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
add_language_in_train_dev_test_files(args)

def remove_file_in_dir(dir_name):
    if os.path.exists(dir_name) and os.path.isdir(dir_name):
        shutil.rmtree(dir_name)

remove_file_in_dir(args.output_dir)
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
write_label_metadata_to_file(args)

