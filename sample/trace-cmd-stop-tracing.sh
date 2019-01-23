#!/bin/bash

pkexec `pwd`/nvgpucs -d -v -m stop

CMD="trace-cmd reset"
echo $CMD
$CMD

CMD="trace-cmd snapshot -f"
echo $CMD
$CMD

./trace-cmd-status.sh
