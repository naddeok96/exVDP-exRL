nvidia-smi -q -x | grep pid | sed -e 's/<pid>//g' -e 's/<\/pid>//g' -e 's/^[[:space:]]*//' | xargs ps -up | awk '/naddeok5/ {print $2}' | xargs kill -9
