#!/bin/bash
#SBATCH --job-name=prep_pretrain         # nom du job
#SBATCH --partition=gpu_p2
# Ici, reservation de 10 CPU (pour 1 tache) et d'un GPU sur un seul noeud :
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
# Le nombre de CPU par tache doit etre adapte en fonction de la partition utilisee. Sachant
# qu'ici on ne reserve qu'un seul GPU (soit 1/4 ou 1/8 des GPU du noeud suivant la partition),
# l'ideal est de reserver 1/4 ou 1/8 des CPU du noeud pour la seule tache:
##SBATCH --cpus-per-task=24 
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU)
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=1:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=prepare_pretrain.out      # nom du fichier de sortie
#SBATCH --error=prepare_pretrain.out       # nom du fichier d'erreur (ici commun avec la sortie)


# Nettoyage des modules charges en interactif et herites par defaut
module purge
 
# Chargement des modules
 
# Echo des commandes lancees
set -x
 
# Pour la partition "gpu_p5", le code doit etre compile avec les modules compatibles
# Execution du code

##/gpfsdswork/dataset/LibriSpeechAsrCorpus/train-other-500/

DATA_LOCATION=/gpfsscratch/rech/bvr/uuk92de/split/
tsv_dir=data/
TRAIN_TEST_RATIO=0.01
feat_dir=feat/
nshard=1
rank=0
CLUSTER_NB=500
KMEAN_SPLIT=0.1
LABEL_LOCATION=label


# Preprocess data
##srun python3 -u fairseq/examples/wav2vec/wav2vec_manifest.py ${DATA_LOCATION} --dest ${tsv_dir} --ext wav --valid-percent ${TRAIN_TEST_RATIO}

# Extract features from 5-th HuBERT-base layer (it-2)
##python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py ${tsv_dir} train hubert_base_ls960.pt 5 ${nshard} ${rank} ${feat_dir}

##python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py ${tsv_dir} valid hubert_base_ls960.pt 5 ${nshard} ${rank} ${feat_dir}

# Generate MFCC
##srun python3 -u fairseq/examples/hubert/simple_kmeans/dump_mfcc_feature.py ${TSV_LOCATION} train ${NSHARD} ${RANK} ${MFCC_LOCATION} 
##srun python3 -u fairseq/examples/hubert/simple_kmeans/dump_mfcc_feature.py ${TSV_LOCATION} valid ${NSHARD} ${RANK} ${MFCC_LOCATION} 

# Compute KMean algorithm to a percentage of the train and valid dataset
# Train/Valid
##srun python3 -u fairseq/examples/hubert/simple_kmeans/learn_kmeans.py ${feat_dir} train ${nshard} km_model ${CLUSTER_NB} --percent ${KMEAN_SPLIT}

##srun python3 -u fairseq/examples/hubert/simple_kmeans/dump_km_label.py ${feat_dir} train km_model ${nshard} ${rank} ${LABEL_LOCATION}

##srun python3 -u fairseq/examples/hubert/simple_kmeans/dump_km_label.py ${feat_dir} valid km_model ${nshard} ${rank} ${LABEL_LOCATION}

#for rank in $(seq 0 $((${nshard} - 1))); do
#  cat ${LABEL_LOCATION}/train_${rank}_${nshard}.km
#done > $(pwd)/${LABEL_LOCATION}/train.km

##for rank in $(seq 0 $((${nshard} - 1))); do
##  cat ${LABEL_LOCATION}/valid_${rank}_${nshard}.km
##done > $(pwd)/${LABEL_LOCATION}/valid.km

# Generate the Dummy Dict (number of clusters)

##for x in $(seq 0 $((${CLUSTER_NB} - 1))); do
##  echo "$x 1"
##done >> ${LABEL_LOCATION}/dict.km.txt