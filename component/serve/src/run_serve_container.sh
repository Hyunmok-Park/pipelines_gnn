#!/bin/sh

cd data
bentoml import simple_gnn.bento
bentoml serve simple_gnn:latest --production
