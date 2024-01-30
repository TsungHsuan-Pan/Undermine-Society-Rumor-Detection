import os
from run import (
    setup_args,
    special_rules,
    set_output_dir_name,
    add_language_in_train_dev_test_files
)
from utils import write_label_text_to_file

args = setup_args()
special_rules(args)
set_output_dir_name(args)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
add_language_in_train_dev_test_files(args)

write_label_text_to_file(args)
