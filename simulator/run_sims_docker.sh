#!/bin/bash

docker rm -f $(docker ps --filter ancestor=lg-simulator -q)

echo Running $1 simulators
for ((p = 0; p < $1; p++)); do
  port=$(expr 8080 + $p)
  docker run -dt -p $port:8080 lg-simulator
done
