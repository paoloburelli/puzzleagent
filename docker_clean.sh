#!/bin/sh
docker rm -f $(docker ps --filter ancestor=lg-simulator -q)