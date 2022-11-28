import os
import sys
import time
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# add path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from datasets.generic import Batch
from models.scoop import SCOOP
from tools.seed import seed_everything
from tools.losses import compute_loss_unsupervised
from tools.utils import log_string


def compute_epe_train(recon_flow, batch):
    """
    Compute EPE during training.

    Parameters
    ----------
    recon_flow: torch.Tensor
        Flow from reconstruction of the target point cloud by the source point cloud.
    batch : scoop.datasets.generic.Batch
        Contains ground truth flow and mask.

    Returns
    -------
    epe : torch.Tensor
        Mean EPE for current batch.

    """

    mask = batch["ground_truth"][0][..., 0]
    true_flow = batch["ground_truth"][1]
    error = recon_flow - true_flow
    error = error[mask > 0]
    epe_per_point = torch.sqrt(torch.sum(torch.pow(error, 2.0), -1))
    epe = epe_per_point.mean()

    return epe


def train(scoop, train_dataloader, val_dataloader, delta, optimizer, scheduler, path2log, args):
    """
    Train scene flow model.

    Parameters
    ----------
    scoop : scoop.models.SCOOP
        SCOOP model
    train_dataloader : scoop.datasets.generic.SceneFlowDataset
        Training Dataset loader.
    val_dataloader : scoop.datasets.generic.SceneFlowDataset
        Validation Dataset loader.
    delta : int
        Frequency of logs in number of iterations.
    optimizer : torch.optim.Optimizer
        Optimiser.
    scheduler :
        Scheduler.
    path2log : str
        Where to save logs / model.
    args : Namespace
        Arguments for training.

    """

    # Set seed
    seed = seed_everything(seed=42)

    # Log directory
    if not os.path.exists(path2log):
        os.makedirs(path2log)
    writer = SummaryWriter(path2log)

    # Log file
    log_file = open(os.path.join(path2log, args.log_fname), 'w')
    log_file.write('Training arguments:')
    log_file.write(str(args) + '\n')
    log_string(log_file, "Seed: %d" % seed)

    # Train
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scoop = scoop.to(device, non_blocking=True)

    total_it = 0
    epoch_start = 0
    for epoch in range(epoch_start, args.nb_epochs):

        # Initialization
        running_loss = 0
        training_loss_tot = 0
        running_target_recon_loss = 0
        training_target_recon_loss_tot = 0
        if args.use_corr_conf:
            running_corr_conf_loss = 0
            training_corr_conf_loss_tot = 0
        if args.use_smooth_flow:
            running_smooth_flow_loss = 0
            training_smooth_flow_loss_tot = 0
        running_epe = 0
        training_epe_tot = 0
        epoch_it = 0

        # Train for one epoch
        start = time.time()
        scoop = scoop.train()
        log_string(log_file, "Training epoch %d out of %d" % (epoch + 1, args.nb_epochs))
        for it, batch in enumerate(tqdm(train_dataloader)):

            # Send data to GPU
            batch = batch.to(device, non_blocking=True)

            # Run model
            recon_flow, corr_conf, target_recon, graph = scoop(batch["sequence"])

            # Compute loss
            loss, target_recon_loss, corr_conf_loss, smooth_flow_loss =\
                compute_loss_unsupervised(recon_flow, corr_conf, target_recon, graph, batch, args)

            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Loss evolution
            loss_curr = loss.item()
            running_loss += loss_curr
            training_loss_tot += loss_curr
            target_recon_loss_curr = target_recon_loss.item()
            running_target_recon_loss += target_recon_loss_curr
            training_target_recon_loss_tot += target_recon_loss_curr
            if args.use_corr_conf:
                corr_conf_loss_curr = corr_conf_loss.item()
                running_corr_conf_loss += corr_conf_loss_curr
                training_corr_conf_loss_tot += corr_conf_loss_curr
            if args.use_smooth_flow:
                smooth_flow_loss_curr = smooth_flow_loss.item()
                running_smooth_flow_loss += smooth_flow_loss_curr
                training_smooth_flow_loss_tot += smooth_flow_loss_curr

            epe_curr = compute_epe_train(recon_flow, batch).item()
            running_epe += epe_curr
            training_epe_tot += epe_curr

            # Logs
            if it % delta == delta - 1:
                # Print / save logs
                writer.add_scalar("Loss/training_loss", running_loss / delta, total_it)
                writer.add_scalar("Loss/training_target_recon_loss", running_target_recon_loss / delta, total_it)
                if args.use_corr_conf:
                    writer.add_scalar("Loss/training_corr_conf_loss", running_corr_conf_loss / delta, total_it)
                if args.use_smooth_flow:
                    writer.add_scalar("Loss/training_smooth_flow_loss", running_smooth_flow_loss / delta, total_it)
                writer.add_scalar("Loss/training_epe", running_epe / delta, total_it)
                print("Epoch {0:d} - It. {1:d}: loss = {2:e}".format(epoch + 1, total_it, running_loss / delta))
                print((time.time() - start) / 60, "minutes")
                # Re-init.
                running_loss = 0
                running_target_recon_loss = 0
                if args.use_corr_conf:
                    running_corr_conf_loss = 0
                if args.use_smooth_flow:
                    running_smooth_flow_loss = 0
                running_epe = 0
                start = time.time()

            epoch_it += 1
            total_it += 1

        # Training loss
        training_loss = training_loss_tot / epoch_it
        writer.add_scalar("Loss/training_loss", training_loss, total_it)
        training_target_recon_loss = training_target_recon_loss_tot / epoch_it
        writer.add_scalar("Loss/training_target_recon_loss", training_target_recon_loss, total_it)
        if args.use_corr_conf:
            training_corr_conf_loss = training_corr_conf_loss_tot / epoch_it
            writer.add_scalar("Loss/training_corr_conf_loss", training_corr_conf_loss, total_it)
        if args.use_smooth_flow:
            training_smooth_flow_loss = training_smooth_flow_loss_tot / epoch_it
            writer.add_scalar("Loss/training_smooth_flow_loss", training_smooth_flow_loss, total_it)
        training_epe = training_epe_tot / epoch_it
        writer.add_scalar("Loss/training_epe", training_epe, total_it)
        log_string(log_file, "Training: loss: %.6f, epe: %.3f" % (training_loss, training_epe))

        # Scheduler
        scheduler.step()

        # Validation
        scoop = scoop.eval()
        val_loss_tot = 0
        val_target_recon_loss_tot = 0
        if args.use_corr_conf:
            val_corr_conf_loss_tot = 0
        if args.use_smooth_flow:
            val_smooth_flow_loss_tot = 0
        val_epe_tot = 0
        val_it = 0
        with torch.no_grad():
            for it, batch in enumerate(tqdm(val_dataloader)):

                # Send data to GPU
                batch = batch.to(device, non_blocking=True)

                # Run model
                recon_flow, corr_conf, target_recon, graph = scoop(batch["sequence"])

                # Compute loss
                loss, target_recon_loss, corr_conf_loss, smooth_flow_loss =\
                    compute_loss_unsupervised(recon_flow, corr_conf, target_recon, graph, batch, args)

                # Validation loss
                val_loss_tot += loss.item()
                val_target_recon_loss_tot += target_recon_loss.item()
                if args.use_corr_conf:
                    val_corr_conf_loss_tot += corr_conf_loss.item()
                if args.use_smooth_flow:
                    val_smooth_flow_loss_tot += smooth_flow_loss.item()
                val_epe_tot += compute_epe_train(recon_flow, batch).item()
                val_it += 1

        val_loss = val_loss_tot / val_it
        writer.add_scalar("Loss/validation_loss", val_loss, total_it)
        val_target_recon_loss = val_target_recon_loss_tot / val_it
        writer.add_scalar("Loss/validation_target_recon_loss", val_target_recon_loss, total_it)
        if args.use_corr_conf:
            val_corr_conf_loss = val_corr_conf_loss_tot / val_it
            writer.add_scalar("Loss/validation_corr_conf_loss", val_corr_conf_loss, total_it)
        if args.use_smooth_flow:
            val_smooth_flow_loss = val_smooth_flow_loss_tot / val_it
            writer.add_scalar("Loss/validation_smooth_flow_loss", val_smooth_flow_loss, total_it)
        val_epe = val_epe_tot / val_it
        writer.add_scalar("Loss/validation_epe", val_epe, total_it)
        log_string(log_file, "Validation: loss: %.6f, epe: %.3f" % (val_loss, val_epe))

        # Save model after each epoch
        state = {
            "args": args,
            "model": scoop.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        if epoch == 0 or (epoch + 1) % args.save_model_epoch == 0:
            suff = args.add_model_suff and "_e%03d" % (epoch + 1) or ""
            torch.save(state, os.path.join(path2log, "model%s.tar" % suff))

    log_string(log_file, "Finished Training.")
    log_file.close()

    return None


def my_main(args):
    """
    Entry point of the script.

    Parameters
    ----------
    args.dataset_name : str
        Dataset for training. Either FlowNet3D_kitti or FlowNet3D_ft3d or HPLFlowNet_kitti or HPLFlowNet_ft3d.
    args.nb_iter : int
        Number of unrolled iteration of Sinkhorn algorithm in SCOOP.
    args.batch_size_train : int
        Batch size fot training dataset.
    args.batch_size_val : int
        Batch size fot validation dataset.
    args.nb_points : int
        Number of points in point clouds.
    args.nb_epochs : int
        Number of epochs.
    args.nb_workers: int
        Number of workers for the dataloader.
    args.log_dir:
        Logging directory.

    Raises
    ------
    ValueError
        If dataset_name is an unknown dataset.

    """

    # Path to current file
    pathroot = os.path.dirname(__file__)

    # Select dataset
    if args.dataset_name.split("_")[0].lower() == "HPLFlowNet".lower():
        # HPLFlowNet version of the datasets
        path2data = os.path.join(pathroot, "..", "data", "HPLFlowNet")

        # KITTI
        if args.dataset_name.split("_")[1].lower() == "kitti".lower():
            path2data = os.path.join(path2data, "KITTI_processed_occ_final")
            from datasets.kitti_hplflownet import Kitti

            # datasets
            train_dataset = Kitti(root_dir=path2data, nb_points=args.nb_points, all_points=False, mode="train")
            val_dataset = Kitti(root_dir=path2data, nb_points=args.nb_points, all_points=False, mode="val")

            # learning rate schedule
            lr_lambda = lambda epoch: 1.0 if epoch < 50 else 1.0

        # FlyingThing3D
        elif args.dataset_name.split("_")[1].lower() == "ft3d".lower():
            path2data = os.path.join(path2data, "FlyingThings3D_subset_processed_35m")
            from datasets.flyingthings3d_hplflownet import FT3D

            # datasets
            train_dataset = FT3D(root_dir=path2data, nb_points=args.nb_points, all_points=False, mode="train", nb_examples=args.nb_train_examples)
            val_dataset = FT3D(root_dir=path2data, nb_points=args.nb_points, all_points=False, mode="val", nb_examples=args.nb_val_examples)

            # learning rate schedule
            lr_lambda = lambda epoch: 1.0 if epoch < 50 else 0.1

        else:
            raise ValueError("Unknown dataset " + args.dataset_name)

    elif args.dataset_name.split("_")[0].lower() == "FlowNet3D".lower():
        # FlowNet3D version of the datasets
        path2data = os.path.join(pathroot, "..", "data", "FlowNet3D")

        # KITTI
        if args.dataset_name.split("_")[1].lower() == "kitti".lower():
            path2data = os.path.join(path2data, "kitti_rm_ground")
            from datasets.kitti_flownet3d import Kitti

            # datasets
            train_dataset = Kitti(root_dir=path2data, nb_points=args.nb_points, all_points=False, same_v_t_split=args.same_val_test_kitti, mode="train")
            val_dataset = Kitti(root_dir=path2data, nb_points=args.nb_points, all_points=False, same_v_t_split=args.same_val_test_kitti, mode="val")

            # learning rate schedule
            lr_lambda = lambda epoch: 1.0 if epoch < 340 else 0.1

        # FlyingThing3D
        elif args.dataset_name.split("_")[1].lower() == "ft3d".lower():
            path2data = os.path.join(path2data, "data_processed_maxcut_35_20k_2k_8192")
            from datasets.flyingthings3d_flownet3d import FT3D

            # datasets
            train_dataset = FT3D(root_dir=path2data, nb_points=args.nb_points, all_points=False, mode="train", nb_examples=args.nb_train_examples)
            val_dataset = FT3D(root_dir=path2data, nb_points=args.nb_points, all_points=False, mode="val", nb_examples=args.nb_val_examples)

            # learning rate schedule
            lr_lambda = lambda epoch: 1.0 if epoch < 340 else 1.0

        else:
            raise ValueError("Unknown dataset" + args.dataset_name)

    else:
        raise ValueError("Invalid dataset name: " + args.dataset_name)

    # Training data
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_train,
        pin_memory=True,
        shuffle=True,
        num_workers=args.nb_workers,
        collate_fn=Batch,
        drop_last=True,
    )

    # Validation data
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_val,
        pin_memory=True,
        shuffle=False,
        num_workers=args.nb_workers,
        collate_fn=Batch,
        drop_last=False,
    )

    # Model
    scoop = SCOOP(args)

    # Optimizer
    optimizer = torch.optim.Adam(scoop.parameters(), lr=args.learning_rate)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Log directory
    path2log = os.path.join(pathroot, "..", "experiments", args.log_dir)


    # Train
    print("Training started. Logs in " + path2log)
    train(scoop, train_dataloader, val_dataloader, 500, optimizer, scheduler, path2log, args)

    return None


if __name__ == "__main__":

    # Args
    parser = argparse.ArgumentParser(description="Train SCOOP.")
    parser.add_argument("--dataset_name", type=str, default="FlowNet3D_kitti", help="Dataset name for train and validation. Either FlowNet3D_kitti or FlowNet3D_ft3d or HPLFlowNet_kitti or HPLFlowNet_ft3d.")
    parser.add_argument("--nb_train_examples", type=int, default=-1, help="Number of examples for the training dataset. active if > 0.")
    parser.add_argument("--nb_val_examples", type=int, default=-1, help="Number of examples for the validation dataset. active if > 0.")
    parser.add_argument("--same_val_test_kitti", type=int, default=1, help="1: Use the same validation and test split for KITTI dataset, 0: Do not use the same validation and test split.")
    parser.add_argument("--batch_size_train", type=int, default=4, help="Batch size fot training dataset.")
    parser.add_argument("--batch_size_val", type=int, default=10, help="Batch size for validation dataset.")
    parser.add_argument("--nb_epochs", type=int, default=400, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--use_corr_conf", type=int, default=1, help="1: Use correspondence confidence for training, 0: Do not use correspondence confidence.")
    parser.add_argument("--linear_corr_conf", type=int, default=0, help="1: Use linear normalization for correspondence confidence, 0: Use ReLU normalization.")
    parser.add_argument("--corr_conf_loss_weight", type=float, default=0.1, help="Weight for correspondence confidence loss.")
    parser.add_argument("--nb_points", type=int, default=2048, help="Maximum number of points in point cloud.")
    parser.add_argument("--nb_iter", type=int, default=1, help="Number of unrolled iterations of the Sinkhorn algorithm.")
    parser.add_argument("--nb_neigh_cross_recon", type=int, default=64, help="Number of neighbor points for cross reconstruction. Active if > 0")
    parser.add_argument("--use_smooth_flow", type=int, default=1, help="1: Use smooth flow loss, 0: Do not use smooth flow loss.")
    parser.add_argument("--nb_neigh_smooth_flow", type=int, default=32, help="Number of neighbor points for smooth flow loss.")
    parser.add_argument("--smooth_flow_loss_weight", type=float, default=10.0, help="Weight for smooth flow loss. Active if > 0.")
    parser.add_argument("--backward_dist_weight", type=float, default=0.0, help="Backward distance weight in Chamfer Distance loss.")
    parser.add_argument("--nb_workers", type=int, default=0, help="Number of workers for the dataloader.")
    parser.add_argument("--log_dir", type=str, default="log_scoop", help="Logging directory.")
    parser.add_argument("--log_fname", type=str, default="log_train.txt", help="Evaluation log file name.")
    parser.add_argument("--save_model_epoch", type=int, default=1, help="Number of epochs difference for saving the model.")
    parser.add_argument("--add_model_suff", type=int, default=0, help="1: Add suffix to model name, 0: do not add suffix.")
    args = parser.parse_args()

    args.use_corr_conf = bool(args.use_corr_conf)
    args.use_smooth_flow = bool(args.use_smooth_flow)
    args.add_model_suff = bool(args.add_model_suff)

    # Check arguments
    if args.use_smooth_flow:
        assert args.nb_neigh_smooth_flow > 0, "If use_smooth_flow is on, nb_neigh_smooth_flow should be > 0 (got %d)." % args.nb_neigh_smooth_flow
        assert args.smooth_flow_loss_weight >= 0, "If use_smooth_flow is on, smooth_flow_loss_weight should be >= 0 (got %f)." % args.smooth_flow_loss_weight

    # Launch training
    my_main(args)
