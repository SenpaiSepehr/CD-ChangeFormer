from argparse import ArgumentParser
import torch
import random
from models.trainer import *
from torch.profiler import profile, record_function, ProfilerActivity

print(torch.cuda.is_available())

"""
the main function for training the CD networks
"""

def seed(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed()

def train(args):
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()
    #model.network_summary(args=args)

def test(args):
    from models.evaluator import CDEvaluator
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models()


def network_summary(args):

    with torch.profiler.profile(
        activities=[ProfilerActivity.CUDA], 
        schedule=torch.profiler.schedule(wait=1,warmup=1,active=3,repeat=2),
        on_trace_ready = torch.profiler.tensorboard_trace_handler('./TB_Oct13/base256_1'),
        record_shapes=True,
        with_stack=True
    ) as p:
        train(args)
        p.stop()

                
            


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='6,7', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='ChangeFormer/DSIFN/base_2.Oct16', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoint', type=str)
    parser.add_argument('--vis_root', default='vis_ChangeMaps', type=str)

    # data
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='DSIFN', type=str)

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--shuffle_AB', default=False, type=str)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--patch_size', default=2, type=int)
    parser.add_argument('--pretrain', default=None, type=str)
    parser.add_argument('--multi_scale_train', default=False, type=str)
    parser.add_argument('--multi_scale_infer', default=False, type=str)
    parser.add_argument('--multi_pred_weights', nargs = '+', type = float, default = [0.5, 0.5, 0.5, 0.8, 1.0])

    parser.add_argument('--net_G', default='ChangeFormerV6', type=str,
                        help='base_resnet18 | base_transformer_pos_s4 | '
                             'base_transformer_pos_s4_dd8 | '
                             'base_transformer_pos_s4_dd8_dedim8|ChangeFormerV5|SiamUnet_diff')
    parser.add_argument('--loss', default='ce', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default=0.00006, type=float)
    parser.add_argument('--max_epochs', default=70, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=100, type=int)

    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)
    
    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    #network_summary(args)
    train(args)
    #test(args)
