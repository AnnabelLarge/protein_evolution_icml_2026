import os
from datetime import datetime

def setup_training_dir(args):
    if 'assert_no_overwrite' not in dir(args):
        args.assert_no_overwrite = True
    
    ### create folder/file names
    tboard_dir = f'{os.getcwd()}/{args.training_wkdir}/tboard/{args.training_wkdir}'
    model_ckpts_dir = f'{os.getcwd()}/{args.training_wkdir}/model_ckpts'
    logfile_dir = f'{os.getcwd()}/{args.training_wkdir}/logfiles'
    out_arrs_dir = f'{os.getcwd()}/{args.training_wkdir}/out_arrs'
    
    # create logfile in the logfile_dir
    logfile_filename = f'PROGRESS.log'
        
    
    ### what to do if training directory exists
    # OPTION 1: IF TRAINING WKDIR ALREAD EXISTS, RAISE RUN TIME ERROR
    if os.path.exists(f'{os.getcwd()}/{args.training_wkdir}') and args.assert_no_overwrite:
        raise RuntimeError(f'{args.training_wkdir} ALREADY EXISTS; DOES IT HAVE DATA?')
    
    # # OPTION 2: IF TRAINING WKDIR ALREADY EXISTS, DELETE IT 
    # elif os.path.exists(f'{os.getcwd()}/{args.training_wkdir}') and not args.assert_no_overwrite:
    #     shutil.rmtree(f'{os.getcwd()}/{args.training_wkdir}')
    
    
    ### make training wkdir and subdirectories
    if not os.path.exists(f'{os.getcwd()}/{args.training_wkdir}'):
        os.mkdir(f'{os.getcwd()}/{args.training_wkdir}')
        os.mkdir(model_ckpts_dir)
        os.mkdir(logfile_dir)
        os.mkdir(out_arrs_dir)
        # tensorboard directory takes care of itself
    
    
    ### add these filenames to the args dictionary, to be passed to training
    ### script
    args.tboard_dir = tboard_dir
    args.model_ckpts_dir = model_ckpts_dir
    args.logfile_dir = logfile_dir
    args.logfile_name = f'{logfile_dir}/{logfile_filename}'
    args.out_arrs_dir = out_arrs_dir
    
    
    ### timestamp
    with open(args.logfile_name, "w") as g:
        g.write(f"Start run on: {datetime.now()}\n\n")
