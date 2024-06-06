# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=False,   #True  # To do quick test. Trains/test on small subset of dataset, and # of epochs
    
       finetune_mode = False,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights = '/mnt/raid/ni/WALE_SEdl/EIN-SELD/Cnn14_DecisionLevelMax_mAP=0.385.pth', 

        # INPUT PATH
        # dataset_dir='DCASE2020_SELD_dataset/foa_eval/',  # Base folder containing the foa/mic and metadata folders
        dataset_dir ='/mnt/raid/ni/wale/DCASE2021_SELD_dataset/', #/scratch/asignal/partha/DCASE2022_SELD_dataset',
                                                                    #'/mnt/raid/ni/wale/DCASE2022_SELD_dataset',
        # OUTPUT PATHS
        # feat_label_dir='DCASE2020_SELD_dataset/feat_label_hnet/',  # Directory to dump extracted features and labels
        feat_label_dir= '/mnt/raid/ni/wale/DCASE2021_SELD_dataset/seld_feat_label', #  '/scratch/asignal/partha/DCASE2022_SELD_dataset/seld_feat_label',
 
        model_dir=  '/mnt/raid/ni/wale/models/',                             #'#/mnt/raid/ni/wale/models/', #'models/',            # Dumps the trained models and training curves in this folder
        dcase_output_dir= '/mnt/raid/ni/wale/results',   ## #'results/',    # recording-wise results are dumped in this path.
                                                                           #'/mnt/raid/ni/wale/results',
        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',       # 'foa' - ambisonic or 'mic' - microphone signals

        #FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins= 128, #64
       
        use_salsalite = False, # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite = 50,
        fmax_doa_salsalite = 2000,
        fmax_spectra_salsalite = 9000,

        # MODEL TYPE
        multi_accdoa= False,#False,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

        # DNN MODEL PARAMETERS
        label_sequence_length= 50,#50,    # Feature sequence length
        batch_size= 128,         # Batch size
        dropout_rate=0.05,             # Dropout rate, constant for all layers
        nb_cnn2d_filt=128,  #64       # Number of CNN nodes, constant for each layer 64
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer 4, 4, 2
        #out_channels=256,
        r= 4,
        context = 128,
        #num_categories = params['unique_classes'],
        nb_rnn_layers=2,
        rnn_size=   128,   # RNN contents, length of list = number of layers, list value = number of nodes
        class_labels= 11,
        self_attn=True, #False
        nb_heads=4,

        nb_fnn_layers=2,#1
        fnn_size=128,             # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=120,              # Train for maximum epochs
        lr=1e-3,

        # METRIC
        average = 'macro',        # Supports 'micro': sample-wise average and 'macro': class-wise average
        lad_doa_thresh=20
    )

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        print("FOA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False

    elif argv == '3':
        print("FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False

    elif argv == '9':
        print("FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
       # params['mode'] = 'eval'
        
        params['use_salsalite'] = False
    elif argv == '4':
        print("MIC + GCC + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False

    elif argv == '5':
        print("MIC + SALSA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False

    elif argv == '6':
        print("MIC + GCC + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True

    elif argv == '7':
        print("MIC + SALSA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True

    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]     # CNN time pooling
    params['patience'] = int(params['nb_epochs'])     # Stop training if patience is reached

    if '2020' in params['dataset_dir']:
        params['unique_classes'] = 14 
    elif '2021' in params['dataset_dir']:
        params['unique_classes'] = 12
    elif '2022' in params['dataset_dir']:
        params['unique_classes'] = 13

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params