from argparse import Namespace
from base.utils import set_random_seed
from base.model import RCEntailModel
import argparse
import pytorch_lightning as pl
import itertools
import os
import torch
import shutil
from base.data.utils import make_few_shots
import json
from base.utils import dispatch_args_to_func
import base

torch.multiprocessing.set_sharing_strategy('file_system')

'''
params = {
    'query_type': [
        "ST1",
    ],
    'event_label_processor': [
        'IdentityEventProcessor',
    ],
    'seed': [0,1,2,3,4],
    'nsamps': [1,3,5,7,9],
    'use_event_description': ['none', 'postfix'],
    'neutral_as': ['negative'],
    'nepochs': [7],
    'max_desc_sentences':[1, 5],
}
'''

def conduct_experiments(model_class, base_path, params, model_kwargs, gpus, monitor='ev_f1', save_top_k=1):

    def zip_all_params(all_params):
        return [{p: q for p, q in zip(all_params.keys(), x)} for x in list(itertools.product(*list(all_params.values())))]

    zipped_params = zip_all_params(params)

    try:
        os.makedirs(base_path)
    except FileExistsError:
        pass

    with open(os.path.join(base_path, 'params.json'), 'w') as f:
        json.dump(params, f)


    results_path = os.path.join(base_path, 'results.csv')

    if not os.path.exists(results_path):
        first_exp = True
    else:
        with open(results_path, 'r') as f:
            if len(f.readlines()) == 1:
                first_exp = True
            else:
                first_exp = False

    if first_exp:
        with open(results_path, 'w') as f:
            f.write(','.join(list(params.keys())))
            f.write(",")

    nexps = len(zipped_params)
    for iarg, args in enumerate(zipped_params):
        exp_name = ''
        for x, y in args.items():
            exp_name += '{}:{}-'.format(x, y)
        if len(exp_name) > 0:
            exp_name = exp_name[:-1]

        exp_path = os.path.join(base_path, exp_name)
        print("Experiment {}/{}".format(
            iarg+1, nexps
        ))
        print("Directory: ")
        print(exp_path)
        try:
            os.makedirs(exp_path)
        except FileExistsError:
            if os.path.exists(os.path.join(exp_path, "FINISHED")):
                print("{} directory exists and has FINISHED flag. Will skip it. ".format(exp_name))
                continue
            else:
                print("{} directory exists but unfinished. Restarting it.".format(exp_name))
                shutil.rmtree(exp_path)
                os.makedirs(exp_path)

        pl.seed_everything(args['seed'])

        model_args = Namespace(**args)
        for x, v in model_kwargs.items():
            setattr(model_args, x, v)

        if type(model_class) is str:
            model = getattr(base.model, model_class)(model_args)
        else:
            model = model_class(model_args)

        trainer = pl.Trainer(
            checkpoint_callback=pl.callbacks.ModelCheckpoint(
                filepath=exp_path,
                verbose=True,
                monitor=monitor,
                mode='max',
                save_top_k=save_top_k,
            ),
            gpus=gpus,
            max_epochs=args['nepochs'],
            distributed_backend='dp',
        )
        trainer.fit(model)
        trainer.test()

        # if first, write callback metrics column names
        with open(results_path, 'a') as f:
            if first_exp:
                first_exp = False
                keys = trainer.callback_metrics.keys()
                f.write(",".join(keys))
                f.write("\n")

            args_str = ",".join([str(x) for x in args.values()])
            results = {x: float(t) if isinstance(t, torch.Tensor) else t 
                        for x, t in trainer.callback_metrics.items()}
            res_str = ",".join([str(x) for x in results.values()])
            f.write(args_str + "," + res_str + "\n")
        with open(os.path.join(exp_path, "FINISHED"), 'w+') as f:
            f.write("")

