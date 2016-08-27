#!/usr/bin/env bash

mysql test -u root -p < query.sql > tweets.txt
