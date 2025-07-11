
if [ $1 -eq 1 ]; then #run ivae on Complete Graph
    python ./experiment_case/Complete_Graph_experimental/ICEVAE_using_our_Synthetic_dataset.py
fi

if [ $1 -eq 2 ]; then #run ivae on Dif_Beta_Exp Complete Graph
    python ./experiment_case/Dif_Beta_Exp/ICEVAE_using_our_Synthetic_dataset_Dif.py
fi

if [ $1 -eq 3 ]; then #run ivae on Equ_con Graph
   python ./experiment_case/equ_con_Assumption_experiment/ICEVAE_using_our_Synthetic_dataset_equ_ver.py
fi

if [ $1 -eq 4 ]; then #run ivae on latent_con Graph
    python ./experiment_case/Susan_Assumption_experiment/ICEVAE_using_our_Synthetic_dataset_without_Z2Y.py
fi


if [ $1 -eq 5 ]; then #run ivae on TWINS
    python ./experiment_case/ICEVAE_using_semi_Synethic_data_TWINs.py
fi

if [ $1 -eq 6 ]; then #run ivae on IHDP
    python ./experiment_case/ICEVAE_using_semi_Synthetic_data_IHDP.py
fi

if [ $1 -eq 7 ]; then #run ivae for MCC
    python ./250113testMCC_PDF/verify_id_of_z_MCC.py
fi
