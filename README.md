# RockIR : Rock Image Retrieval
This repo covers content-based image retrieval for industrial earth material images (here: rock images) to resolve data paucity, “expert knowledge” subjectivity, and uncommon statistical image properties. Since the lower number of data in data sets and generating labels by human experts is too costly and work intensive for rock samples, a data efficient training algorithm was built here.

To resolve the data paucity issue, pretrained DL models are used in conjunction with “few-shot” transfer learning. For this, a double Siamese neural network (DSNN) is proposed with ResNet-18 as backbone networks. Rather than relying on expert subjectivity of image similarity, the metadata from physical measurements are used for training. In addition to the image based similarity, metadata based similarity is included in the loss function as well as regression term for predicting metadata. The DSNN outperforms Vanilla ResNet-34, Siamese ResNet-18, Siamese ResNet-34 in image retrieval. Also, qualitative performance comparison is made to find how DSNN retrieves similar samples in a database comparing to baseline Vanilla ResNet-34. 

Given its improved performance on retrieval to find similar samples in terms of both image and metadata, the proposed Double Siamese Neural Network can be done in seconds, compared to weeks or months required by approaches using computer simulations or physical experimentation on rock samples. 

## Getting started
1. Make sure that you have essential packages to run these codes based on requirements.txt 
2. Clone this repo for input images 
3. Run a following command for training a network with 4 gpus.
    ```sh
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=1111 ./src/main_siamese_rock18_gated_ddp.py --datapath='your_data_path' --metadata_path='your_metadata_path'
    ```
4. Or, if you want to run with a gpu, 
    ```sh
    python ./src/main_without_DDP.py --datapath='your_data_path' --metadata_path='your_metadata_path'
    ```
5. To test gated based network with top-3 retrievals on same sample, same reservoir, and different reservoir.
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
Distributed under the MIT License. See `LICENSE.md` for more information.

## Contribution
It would be great if you could contribute to this project. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

## Acknowledgments
We would like to acknowledge 

* Sherin Mirza, Aarthi Thyagarajan and Luud Heck from Shell supporting the OpenSource release on GitHub 

## How to Cite
Myung Seok Shim, Christopher Thiele, Jeremy Vila, Nishank Saxena, Detlef Hohl, Content-based image retrieval for industrial material images with deep learning and encoded physical properties, Data-Centric Engineering, Volume 4, 2023, e21, DOI: [https://doi.org/10.1017/dce.2023.16](https://doi.org/10.1017/dce.2023.16)
