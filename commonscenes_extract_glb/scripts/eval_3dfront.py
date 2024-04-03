from __future__ import print_function

import time

import open3d as o3d # open3d needs to be imported before other packages!
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from model.VAE import VAE
from dataset.threedfront_dataset import ThreedFrontDatasetSceneGraph
from helpers.util import bool_flag, batch_torch_denormalize_box_params, sample_points
from helpers.metrics_3dfront import validate_constrains, validate_constrains_changes, estimate_angular_std
from helpers.visualize_scene import render, render_v2_full, render_v2_box, render_v1_full

import extension.dist_chamfer as ext
chamfer = ext.chamferDist()
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--num_points', type=int, default=1024, help='number of points in the shape')
parser.add_argument('--num_samples', type=int, default=3, help='for diversity')

parser.add_argument('--dataset', required=False, type=str, default="../data/FRONT", help="dataset path")
parser.add_argument('--with_points', type=bool_flag, default=False, help="if false, only predicts layout")
parser.add_argument('--with_feats', type=bool_flag, default=False, help="Load Feats directly instead of points.")
parser.add_argument('--with_CLIP', type=bool_flag, default=True, help="Load Feats directly instead of points.")

parser.add_argument('--manipulate', default=True, type=bool_flag)
parser.add_argument('--path2atlas', default="../experiments/atlasnet/model_70.pth", type=str)
parser.add_argument('--exp', default='../experiments/layout_test', help='experiment name')
parser.add_argument('--epoch', type=str, default='100', help='saved epoch')
parser.add_argument('--recompute_stats', type=bool_flag, default=False, help='Recomputes statistics of evaluated networks')
parser.add_argument('--evaluate_diversity', type=bool_flag, default=False, help='Computes diversity based on multiple predictions')
parser.add_argument('--gen_shape', default=False, type=bool_flag, help='infer diffusion')
parser.add_argument('--visualize', default=False, type=bool_flag)
parser.add_argument('--export_3d', default=False, type=bool_flag, help='Export the generated shapes and boxes in json files for future use')
parser.add_argument('--no_stool', default=False, type=bool_flag)
parser.add_argument('--room_type', default='all', help='all, bedroom, livingroom, diningroom, library')

args = parser.parse_args()

os.environ['PYOPENGL_PLATFORM'] = 'egl'

room_type = ['all', 'bedroom', 'livingroom', 'diningroom', 'library']


def reseed(num):
    np.random.seed(num)
    torch.manual_seed(num)
    random.seed(num)

def evaluate():
    print(torch.__version__)

    random.seed(48)
    torch.manual_seed(48)

    argsJson = os.path.join(args.exp, 'args.json')

    assert os.path.exists(argsJson), 'Could not find args.json for experiment {}'.format(args.exp)

    with open(argsJson) as j:
        modelArgs = json.load(j)

    normalized_file = os.path.join(args.dataset, 'boxes_centered_stats_{}_trainval.txt').format(modelArgs['room_type'])

    # used to collect train statistics
    stats_dataset = ThreedFrontDatasetSceneGraph(
        root=args.dataset,
        split='train_scans',
        use_scene_rels=modelArgs['use_scene_rels'],
        with_changes=False,
        eval=False,
        with_feats=modelArgs['with_feats'],
        with_CLIP=modelArgs['with_CLIP'],
        use_SDF=False,
        large=modelArgs['large'],
        room_type=modelArgs['room_type'])

    test_dataset_no_changes = ThreedFrontDatasetSceneGraph(
        root=args.dataset,
        split='val_scans',
        use_scene_rels=modelArgs['use_scene_rels'],
        with_changes=False,
        eval=True,
        with_feats=modelArgs['with_feats'],
        with_CLIP=modelArgs['with_CLIP'],
        use_SDF=modelArgs['with_SDF'],
        large=modelArgs['large'],
        room_type=args.room_type)

    if args.with_points:
        # collate_fn1 = test_dataset_rels_changes.collate_fn_vaegan_points
        # collate_fn2 = test_dataset_addition_changes.collate_fn_vaegan_points
        collate_fn3 = stats_dataset.collate_fn_vaegan_points
        collate_fn4 = test_dataset_no_changes.collate_fn_vaegan_points
    else:
        # collate_fn1 = test_dataset_rels_changes.collate_fn_vaegan
        # collate_fn2 = test_dataset_addition_changes.collate_fn_vaegan
        collate_fn3 = stats_dataset.collate_fn_vaegan
        collate_fn4 = test_dataset_no_changes.collate_fn_vaegan

    # dataloader to collect train data statistics
    stats_dataloader = torch.utils.data.DataLoader(
        stats_dataset,
        batch_size=1,
        collate_fn=collate_fn3,
        shuffle=False,
        num_workers=0)

    test_dataloader_no_changes = torch.utils.data.DataLoader(
        test_dataset_no_changes,
        batch_size=1,
        collate_fn=collate_fn4,
        shuffle=True,
        num_workers=0)

    modeltype_ = modelArgs['network_type']
    replacelatent_ = modelArgs['replace_latent'] if 'replace_latent' in modelArgs else None
    with_changes_ = modelArgs['with_changes'] if 'with_changes' in modelArgs else None
    modelArgs['no_stool'] = args.no_stool if 'no_stool' not in modelArgs else modelArgs['no_stool']
    diff_opt = modelArgs['diff_yaml'] if modeltype_ == 'v2_full' else None
    try:
        with_E2 = modelArgs['with_E2']
    except:
        with_E2 = True

    model = VAE(root=args.dataset, type=modeltype_, diff_opt=diff_opt, vocab=test_dataset_no_changes.vocab, replace_latent=replacelatent_,
                with_changes=with_changes_, residual=modelArgs['residual'], gconv_pooling=modelArgs['pooling'], clip=modelArgs['with_CLIP'],
                with_angles=modelArgs['with_angles'],deepsdf=modelArgs['with_feats'],  with_E2=with_E2)
    if modeltype_ == 'v2_full':
        # args.visualize = False if args.gen_shape==False else args.visualize
        model.vae_v2.optimizer_ini()
    model.load_networks(exp=args.exp, epoch=args.epoch, restart_optim=False)
    if torch.cuda.is_available():
        model = model.cuda()

    model = model.eval()
    model.compute_statistics(exp=args.exp, epoch=args.epoch, stats_dataloader=stats_dataloader, force=args.recompute_stats)
    print("calculated mu and sigma")

    cat2objs = None


    print('\nEditing Mode - Additions')
    reseed(47)
    # validate_constrains_loop_w_changes(modelArgs, test_dataloader_add_changes, model, normalized_file=normalized_file, with_diversity=args.evaluate_diversity, with_angles=modelArgs['with_angles'], num_samples=args.num_samples, cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

    reseed(47)
    print('\nEditing Mode - Relationship changes')
    # validate_constrains_loop_w_changes(modelArgs, test_dataloader_rels_changes, model,  normalized_file=normalized_file, with_diversity=args.evaluate_diversity, with_angles=modelArgs['with_angles'], num_samples=args.num_samples, cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

    reseed(47)
    print('\nGeneration Mode')
    validate_constrains_loop(modelArgs, test_dataloader_no_changes, model, epoch=args.epoch, normalized_file=normalized_file, with_diversity=args.evaluate_diversity,
                             with_angles=modelArgs['with_angles'], num_samples=args.num_samples, vocab=test_dataset_no_changes.vocab,
                             point_classes_idx=test_dataset_no_changes.point_classes_idx,
                             export_3d=args.export_3d, cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

def validate_constrains_loop(modelArgs, testdataloader, model, epoch=None, normalized_file=None, with_diversity=True, with_angles=False, vocab=None,
                             point_classes_idx=None, export_3d=False, cat2objs=None, datasize='large',
                             num_samples=3, gen_shape=False):

    if with_diversity and num_samples < 2:
        raise ValueError('Diversity requires at least two runs (i.e. num_samples > 1).')

    accuracy = {}
    for k in ['left', 'right', 'front', 'behind', 'smaller', 'bigger', 'shorter', 'taller', 'standing on', 'close by', 'symmetrical to', 'total']:
        # compute validation for these relation categories
        accuracy[k] = []

    all_diversity_boxes = []
    all_diversity_angles = []
    all_diversity_chamfer = []

    bed_diversity_chamfer = []
    night_diversity_chamfer = []
    wardrobe_diversity_chamfer = []
    chair_diversity_chamfer = []
    table_diversity_chamfer = []
    cabinet_diversity_chamfer = []
    sofa_diversity_chamfer = []
    lamp_diversity_chamfer = []
    shelf_diversity_chamfer = []
    tvstand_diversity_chamfer = []


    all_pred_shapes_exp = {} # for export
    all_pred_boxes_exp = {}
    bbox_file = "../data/FRONT/cat_jid_trainval.json" if datasize == 'large' else "../data/FRONT/cat_jid_trainval_small.json"

    from model.diff_utils.util_3d import init_mesh_renderer
    dist, elev, azim = 1.7, 20, 20
    sdf_renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim,device='cuda')
    with open(bbox_file, "r") as read_file:
        box_data = json.load(read_file)
        if modelArgs['no_stool']:
            box_data['chair'].update(box_data['stool'])
    pbar = tqdm(testdataloader, total=len(testdataloader))
    for i, data in enumerate(pbar, 0):
        pbar.set_description(f"Scene_id: {data['scan_id'][0]}")
        try:
            dec_objs, dec_triples = data['decoder']['objs'], data['decoder']['tripltes']
            instances = data['instance_id'][0]
            scan = data['scan_id'][0]
        except Exception as e:
            print(e)
            continue

        dec_objs, dec_triples = dec_objs.cuda(), dec_triples.cuda()
        encoded_dec_text_feat, encoded_dec_rel_feat = None, None
        if modelArgs['with_CLIP']:
            encoded_dec_text_feat, encoded_dec_rel_feat = data['decoder']['text_feats'].cuda(), data['decoder']['rel_feats'].cuda()
        dec_sdfs = None
        if modelArgs['with_SDF']:
            dec_sdfs = data['decoder']['sdfs']

        all_pred_boxes = []

        with torch.no_grad():

            boxes_pred, shapes_pred = model.sample_box_and_shape(point_classes_idx, dec_objs, dec_triples, dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat, attributes=None, gen_shape=gen_shape)
            if with_angles:
                boxes_pred, angles_pred = boxes_pred
                angles_pred = -180 + (torch.argmax(angles_pred, dim=1, keepdim=True) + 1)* 15.0 # TODO angle (previously minus 1, now add it back)
            else:
                angles_pred = None

            # if model.type_ != 'v2_box' and model.type_ != 'dis' and model.type_ != 'v2_full':
            #     shapes_pred, shape_enc_pred = shapes_pred

            if model.type_ == 'v1_full':
                shape_enc_pred = shapes_pred
                #TODO Complete shared shape decoding

                shapes_pred, _ = model.decode_g2sv1(dec_objs, shape_enc_pred, box_data, retrieval=True)

        boxes_pred_den = batch_torch_denormalize_box_params(boxes_pred,file=normalized_file)

        if args.visualize:
            colors = None
            classes = sorted(list(set(vocab['object_idx_to_name'])))
            # layout and shape visualization through open3d
            if model.type_ == 'v1_box' or model.type_ == 'v2_box':
                render_v2_box(model.type_, data['scan_id'], dec_objs.detach().cpu().numpy(), boxes_pred_den, angles_pred, datasize=datasize, classes=classes, render_type='retrieval',
                       classed_idx=dec_objs, store_img=True, render_boxes=False, visual=True, demo=False, no_stool = args.no_stool, without_lamp=True)
            elif model.type_ == 'v1_full':
                render_v1_full(model.type_, data['scan_id'], dec_objs.detach().cpu().numpy(), boxes_pred_den, angles_pred, datasize=datasize, classes=classes, render_type='v1', classed_idx=dec_objs,
                    shapes_pred=shapes_pred, store_img=True, render_boxes=False, visual=False, demo=False, no_stool = args.no_stool, without_lamp=True)
            elif model.type_ == 'v2_full':
                if shapes_pred is not None:
                    shapes_pred = shapes_pred.cpu().detach()
                render_v2_full(model.type_, data['scan_id'], dec_objs.detach().cpu().numpy(), boxes_pred_den, angles_pred, classes=classes, render_type='v2', classed_idx=dec_objs,
                    shapes_pred=shapes_pred, store_img=True, render_boxes=False, visual=False, demo=False,epoch=epoch, without_lamp=False)

        all_pred_boxes.append(boxes_pred_den.cpu().detach())

    if export_3d:
        # export box and shape predictions for future evaluation
        result_path = os.path.join(args.exp, 'results')
        if not os.path.exists(result_path):
            # Create a new directory for results
            os.makedirs(result_path)
        shape_filename = os.path.join(result_path, 'shapes_' + ('large' if datasize else 'small') + '.json')
        box_filename = os.path.join(result_path, 'boxes_' + ('large' if datasize else 'small') + '.json')
        json.dump(all_pred_boxes_exp, open(box_filename, 'w')) # 'dis_nomani_boxes_large.json'
        json.dump(all_pred_shapes_exp, open(shape_filename, 'w'))



if __name__ == "__main__":
    evaluate()
