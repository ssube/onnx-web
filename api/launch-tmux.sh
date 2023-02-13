#! /bin/sh

tmux new-session -d -s onnx-web
tmux send-keys -t onnx-web './launch.sh' 'C-m'
tmux split-window -t onnx-web -v 'watch nvidia-smi'
tmux split-window -t onnx-web -h 'htop'
tmux attach -t onnx-web
