#!/bin/bash
## Readme
# Run this script under /src
## Readme __eof__

## Set env parameter
## Add CookDial/src to python path
export PYTHONPATH=$PYTHONPATH:`pwd`/..


PROGNAME=$(basename "$0")
help_info() {
    echo "${PROGNAME}: usage: ${PROGNAME}"$'\n'\
          "[-c/--config (required) the config file "$'\n'\
          "[-m/--mode (required) train or test "$'\n'\
          "[-r/--resume (optional) the path to checkpoint "$'\n'\
          "-h/--help]"
    return
}

process_arguments() {
    while [[ -n "$1" ]]; do
        case "$1" in
            -c | --config)  shift  # the affix of config file (user_task, agent_task, gene_task)
                            config_file="$1"
                            ;;
            -m | --mode)  shift
                            mode="$1"
                            ;;
            -r | --resume)  shift  #
                            resume_path="${1:-'na'}"
                            ;;
            -h | --help)    help_info
                            exit
                            ;;
            *)              help_info
                            exit 1
                            ;;
        esac
        shift
    done
}

process_arguments "$@"

if [[ ${mode} == "train" ]]; then
  python train.py -c "${config_file}" --mode "train"
else
  if [[ -e ${resume_path} ]]; then
    python train.py -c "${config_file}" --mode "test" --resume "${resume_path}"
  else
    echo "Wrong resume path!!"
    exit 1
  fi
fi

if [ $? -eq 0 ]
then
  exit 0
else
  # Redirect stdout from echo command to stderr.
  echo "Script exited with error." >&2
  exit 1
fi