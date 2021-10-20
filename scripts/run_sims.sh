#!/bin/bash

killall linux.x86_64
echo Running $1 simulators
for ((p = 0; p < $1; p++)); do
  port=$(expr 8080 + $p)
  simulator/linux.x86_64 -port $port &
done
