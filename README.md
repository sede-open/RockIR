# RockIR : Rock Image Retrieval
This project is to build content-based image retrieval for 2D Micro-CT rock images based on Python.

## Purpose
Analogue rock image search of 2D Micro-CT images

## Getting started
1. Make sure that you have essential packages to run these codes based on requirements.txt 
2. Clone ths repo for input images 
3. Run a following command for training a network with 4 gpus.
    ```sh
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=1111 ./src/main_siamese_rock18_gated_ddp.py --datapath='your_data_path' --metadata_path='your_metadata_path'
    ```
4. Or, if you want to run with a gpu, 
    ```sh
    python ./src/main_without_DDP.py --datapath='your_data_path' --metadata_path='your_metadata_path'
    ```
5. To test gated based network with top-3 retrievals on same sample, same reservoir, different reservoir, and bottom-3
    ```sh
    python ./tests/cbir_rock18_gated_qualitative_analysis.py --db_path='your_database_path' --metadata_path='your_metadata_path'
    ```

## Dataset path structure
1. For training set : ```./data/yourdataset/train ```
2. For val set : ```./data/yourdataset/val ```
3. For image retrieval test : ```./data/yourdataset/db_test ```
4. For metadata : ```./data/yourdataset/yourmetadata.csv ```

## Support and contacts
Contact : Myung-Seok(David) Shim (Myung-Seok.Shim@shell.com)

## Licensing
If there is anything special about licensing, mention/reference it here. The default copyright notice is in [LICENSE](LICENSE).

## Contribution
For support and contributing to the project, refer to [CONTRIBUTING.md](CONTRIBUTING.md).
