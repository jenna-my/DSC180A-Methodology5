#!/usr/bin/env python

import sys
import os
import json
import pandas as pd

sys.path.insert(0, 'src')

import env_setup
from etl import get_data
from features import apply_features

from model import model_build


def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    `main` runs the targets in order of data=>analysis=>model.
    '''

    env_setup.make_datadir()
    # env_setup.auth()

    if 'data' in targets:
        # with open('config/data-params.json') as fh:
        #     data_cfg = json.load(fh)
        # # make the data target
        data = pd.read_csv('test/testdata/test_data.csv')

    if 'features' in targets:
        # with open('config/features-params.json') as fh:
        #     feats_cfg = json.load(fh)

        feats, labels = apply_features(data)

    if 'model' in targets:
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)
        model_build(feats, labels, **model_cfg)

    if 'test' in targets:
        # Data target - test data
        data = pd.read_csv('test/testdata/test_data.csv')

        # Feature target
        feats, labels = apply_features(data)

        # Model target
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)
        model_build(feats, labels, **model_cfg)
    return


if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)