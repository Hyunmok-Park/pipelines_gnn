#!/bin/bash

BENTO=$1
bentoml import BENTO
bentoml serve simple_gnn:latest --production