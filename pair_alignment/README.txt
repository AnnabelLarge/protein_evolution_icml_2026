Pair alignment: training code
===============================
JAX/Flax framework for training probabilistic pairwise sequence alignment
models. Supports classical pairHMMs and neural network-augmented variants.


Requirements
-------------
See conda_env.yaml for the full environment. Key packages:

    Python   3.9.18
    JAX      0.4.28
    Flax     0.8.3
    PyTorch  2.2.2   (dataloaders and TensorBoard only)

To recreate the environment:

    conda env create -f conda_env.yaml
    conda activate env

Requires a single GPU. Before running, set:

    export CUDA_VISIBLE_DEVICES=0


Directory structure
--------------------

Run from the project root: python pair_alignment -task ... -configs ...

__main__.py                     Entry point; dispatches on -task and -configs
conda_env.yaml                  Conda environment specification
example_data.zip                Example data (unzip before use)
example_configs.zip             Example JSON configs that train on example data

cli/                            Per-model CLI wrappers
    train_<model>.py            Fresh training
    cont_training_<model>.py    Resume training from a checkpoint
    eval_<model>.py             Evaluation

dloaders/                       Data pipeline
    CountsDset.py               Dataset of precomputed alignment summary statistics
    FullLenDset.py              Dataset of full-length alignment matrices
    init_*.py                   Dataset / dataloader initializers
    init_time_array.py          Branch-length array construction

latent_class_mixtures/          Classical pairHMM models
    IndpSites.py                Standard pair HMMs and mixture of site classes, 
                                mixSites
    FragAndSiteClasses.py       Mixture of fragment classes, mixFrag
    NestedTKF.py                Mixture of domain classes, mixDom
    emission_models.py          GTR-LG08 / F81 substitution models
    transition_models.py        TKF91 / TKF92 / geometric-length transitions
    model_functions.py          TKF math, matrix exponentials, likelihood scoring

neural_models/                  Neural network models
    sequence_embedders/
        cnn/                         Code for CNN sequence embedder
        lstm/                        Code for LSTM sequence embedder
        transformer/                 Code for Transformer sequence embedder
        concatenation_fns.py         Gathers position-specific embeddings
        initial_embedding_blocks.py  Initial embedding blocks
    neural_hmm_predict/
        NeuralCondTKF.py        Neural TKF prediction head
        emission_models.py      Local equilibrium and F81 modules
        transition_models.py    Local TKF92 module
        scoring_fns.py          Substitution and indel log-likelihood scoring
        model_functions.py      TKF math (alpha/beta/gamma, small-time approximations)
    feedforward_predict/
        FeedforwardPredict.py   Basic neural prediction head
    neural_shared/
        neural_initializer.py     Initializes all three Flax Trainstate objects
        postprocessing_models.py  Modules to process column-specific embeddings

train_eval_fns/                 Training and evaluation functions
    general_training_wrapper/
        TrainingWrapper.py           Base class + 4 subclasses (NeuralTKF, Feedforward,
                                     TransitMixes, IndpSites); epoch loop, early stopping,
                                     checkpointing, NaN detection
        batch_metric_helpers.py      Records gradients, Adam states, intermediates
        training_wrapper_helpers.py  Timers, bin-clipping, metric aggregation
    neural_hmm_predict_train_eval_one_batch.py    training wrapper for Neural TKF
    feedforward_predict_train_eval_one_batch.py   training wrapper for Basic neural
    indp_site_classes_training_fns.py             training and evaluation wrappers for 
                                                  basic pair HMMs + mixSites
    transit_mixes_training_fns.py                 training and evaluation wrappers for 
                                                  mixFrag and mixDom
    neural_final_eval_wrapper.py                  evaluation wrapper for all neural models

utils/
    BaseClasses.py              ModuleBase, neuralTKFModuleBase
    build_optimizer.py          AdamW, optional warmup-cosine LR, gradient accumulation
    edit_argparse.py            Default-filling for config entries
    setup_training_dir.py       Creates output directory structure
    end_training.py             Writes final eval results
    write_config.py             Serializes argparse to JSON


Model types
------------

Set pred_model_type in your config to one of:

  pairhmm_indp_sites, pairhmm_indp_sites_old_style          
                              Basic pair hmms, mixSites
                              Uses precomputed alignment counts (CountsDset).

  pairhmm_frag_and_site_classes
                              mixFrag
                              Uses full-length alignment matrices (FullLenDset).

  pairhmm_nested_tkf          mixDom
                              Uses full-length alignment matrices (FullLenDset).

  neural_hmm                  Neural TKF.
                              Uses full-length alignment matrices (FullLenDset).

  feedforward                 Basic neural.
                              Uses full-length alignment matrices (FullLenDset).


How to run
-----------

All commands are run from the project root directory.

Unzip example data and configs first:

    unzip pair_alignment/example_data.zip -d pair_alignment/
    unzip pair_alignment/example_configs.zip -d pair_alignment/

--- Train from scratch ---

    CUDA_VISIBLE_DEVICES=0 python pair_alignment \
        -task train \
        -configs path/to/config.json

--- Resume training ---

    CUDA_VISIBLE_DEVICES=0 python pair_alignment \
        -task continue_train \
        -configs path/to/original_config.json \
        -new_training_wkdir NEW_RUN_NAME \
        -prev_model_ckpts_dir path/to/previous/model_ckpts \
        -tstate_to_load CHECKPOINT_SUFFIX

--- Evaluate ---

    CUDA_VISIBLE_DEVICES=0 python pair_alignment \
        -task eval \
        -configs path/to/eval_config.json

The eval config must include training_wkdir pointing to the original training
output directory; the training argparse is read automatically from there.


Config files
-------------

Configs are JSON files. The example_configs/ directory has working examples
for each model type — the easiest starting point is to copy one and modify it.

Key top-level fields:

    pred_model_type         One of the five model types above
    training_wkdir          Output directory name
    num_epochs              Number of training epochs

    pred_config             Dict of model-specific settings, including:
        times_from          "geometric" | "t_array_from_file" | "t_per_sample"
        indel_model_type    "tkf91" | "tkf92"  (neural_hmm only)

    optimizer_config        Dict with:
        init_value          Initial LR
        peak_value          Peak LR (after warmup)
        end_value           Final LR (after cosine decay)
        warmup_steps        Number of warmup steps
        weight_decay        AdamW weight decay
        every_k_schedule    Gradient accumulation steps (optax.MultiSteps)

For full documentation of all config entries, see ./config_entries/.


OUTPUTS
=======

All outputs are written under training_wkdir/:

    tboard/                 TensorBoard event files
    model_ckpts/            Model state dicts (pickle), saved periodically
                            and at best dev-set loss
    logfiles/               Per-sample loss TSVs, timing tables, flat loss
                            trajectory (losses_flat.tsv)
    params/                 Final parameter arrays as .npy files


Conventions
------------

Alphabet       20 amino acids + <bos> + <eos>
Padding        Sequences padded with 0; alignment indices padded with -9;
               gap token = 43
Alignment      Match=1, Insert=2, Delete=3, START=4, END=5
Time modes     geometric / t_array_from_file: grid of times, marginalized with
               an exponential prior; t_per_sample: one branch length per pair
               
