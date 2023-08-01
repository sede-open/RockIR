import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glob import glob

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

import dataloader.dataset_cbir as dataset
from src import parser_define


output_size = 512
train_args = parser_define.train_args_define()
db_image_path = train_args.db_path # image_paths should be fixed 
image_paths = glob(train_args.db_path+'/*') # put all imgs in the path into a list
metadata_path = train_args.metadata_path # metadata_path


def image_name_extractor(image_path_filename):
    # Remove unnecessary parts of image name
    counter_underscore = 0
    for l in range(len(image_path_filename)):
        if (image_path_filename[l] == "_") :
            counter_underscore += 1
            if (counter_underscore == 1):
                rocksample_name = image_path_filename[:l]
                # print (rocksample_name)
                break
    
    return rocksample_name


def ap_k(actual, predicted, k):
    relevant_counter = 0
    precisions = np.zeros(k)
    
    for i in range(len(predicted)):
        if (relevant_counter < k): 
            if (predicted[i] == actual) :
                relevant_counter += 1
                precisions[relevant_counter-1]=float(relevant_counter)/(i+1)
                
    ap = np.mean(precisions)

    return ap


def inference(model_resnet,  test_loader, args):
    print('running inference for database')
    model_resnet.eval()

    predictions = np.zeros((len(test_loader), output_size))
    
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            # print ('CBIRDatabaseClusters: Computing prediction for {}th image'.format(i+1))
            img, label = data
            img = img.to(args.device)

            output = model_resnet(img)

            # Normalization
            output = F.normalize(output, dim=1)
            
            # Save prediction to list
            predictions[i] = torch.reshape(output, (output_size, )).cpu().numpy()

    return predictions


def query(model_resnet,img, num, predictions, actual):
    model_resnet.eval()

    # Utilize the neural network to the query image
    output = model_resnet(img)

    # Normalization
    output = F.normalize(output, dim=1)
    prediction = output

    # print(prediction.shape)
    prediction = torch.reshape(prediction, (-1, 1)).cpu().detach().numpy().flatten()

    index = faiss.IndexFlatL2(output_size)
    predictions = predictions.astype('float32')
    index.add(predictions)

    distance, indices = index.search(prediction.reshape(1, output_size), num)

    imgIDs = indices[0]
    scores = np.sqrt(distance[0] / output_size)

    predicted = []

    for i in imgIDs:    
        img_path = pathlib.Path(image_paths[i])
        predicted.append(img_path.stem[:2])

    # MAP@10
    apk = ap_k(actual, predicted, 10)
    # print ("Average Precision@n:", apk)

    return imgIDs, scores, apk


#1st row : top-3 retrieval from same sample, 2nd row : top-3 retrieval from same reservoir, 3rd row : top-3 retrieval from different reservoir
def showResults(imgIDs):
    fig, axs = plt.subplots(3, 3, constrained_layout=True)

    query_filename = pathlib.Path(image_paths[imgIDs[0]]).stem
    try:
        df_metadata = pd.read_csv(metadata_path)
    except:
        df_metadata = pd.DataFrame(columns=['Image', 'Label', 'Porosity', 'Permeability', 'Pixelsize'])
        pass

    same_sample = []
    same_res = []
    diff_res = []
    abs_rank_sample = []
    abs_rank_res = []
    abs_rank_diff_res = []

    sample_query = query_filename[:6]
    res_query = query_filename[:3]
    for i in range(len(imgIDs)):
        if pathlib.Path(image_paths[imgIDs[i]]).stem[:6] == sample_query:
            same_sample.append(imgIDs[i])
            abs_rank_sample.append(i)

    for i in range(len(imgIDs)):       
        if pathlib.Path(image_paths[imgIDs[i]]).stem[:3] == res_query:
            if pathlib.Path(image_paths[imgIDs[i]]).stem[:6] != sample_query:
                same_res.append(imgIDs[i])
                abs_rank_res.append(i)
        else:
            diff_res.append(imgIDs[i])      
            abs_rank_diff_res.append(i)

    if len(same_res)==0:
        same_res = same_sample
        abs_rank_res = abs_rank_sample

    same_res = [x for _,x in sorted(zip(abs_rank_res,same_res))]
    abs_rank_res = sorted(abs_rank_res)

    diff_res = [x for _,x in sorted(zip(abs_rank_diff_res,diff_res))]
    abs_rank_diff_res = sorted(abs_rank_diff_res)

    list_index = same_sample[0:3] + same_res[0:3] + diff_res[0:3]  # [0,1,2,3,4,5,6,7,8,-3,-2,-1]
    rank_index = abs_rank_sample[0:3] + abs_rank_res[0:3] + abs_rank_diff_res[0:3] 
    
    print("list for 3x3 results:", list_index)
    print("ranking for 3x3 results:", rank_index)

    for j in range(0, 3):
        for k in range(0, 3):
            index_image = list_index[3*j+k]
            img = Image.open(image_paths[index_image]).convert('LA')
            axs[j, k].imshow(img)

            found_microct_df = df_metadata[df_metadata['Image'].str.match(image_name_extractor(pathlib.Path(image_paths[index_image]).stem))]

            por = found_microct_df['Porosity'].values
            perm = found_microct_df['Permeability'].values

            axs[j, k].set_title(
                "{}\npor:{}, perm:{} \n ABS ranking:{}/{}".format(pathlib.Path(image_paths[index_image]).stem, por, perm, 1+rank_index[3*j + k], len(imgIDs)), pad=-20, fontsize=8)
            
            axs[j, k].axis('off')

    plt.show()

    
def main_func():
    train_args = parser_define.train_args_define()

    torch.manual_seed(train_args.seed)
    train_args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('current device', train_args.device)

    # Setting a MicroCT dataset
    microct_dataset = dataset.Microct_Dataset(image_paths=db_image_path, 
                                            transform=transforms.Compose([
                                                transforms.Resize((224, 224)),
                                                transforms.CenterCrop((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (0.5,))
                                                ]))

    test_loader = torch.utils.data.DataLoader(
        microct_dataset,
        batch_size=1, 
        shuffle=False)

    train_args.test_loader = test_loader

    # Load a resnet architecture
    model_resnet = models.resnet34(pretrained=True) 
    model_resnet = torch.nn.Sequential(*(list(model_resnet.children())[:-1])) # Remove last fc layer
    model_resnet.to(train_args.device)

    train_args.test_loader = test_loader
    predictions  = inference(model_resnet, train_args.test_loader, train_args)

    # Test image index from db
    index_image = 3

    # Query function
    num = len(image_paths)

    apks = np.zeros(len(test_loader))
    for i, data in enumerate(test_loader, 0):
        if (i == index_image):
            img, label = data
            input_img = img.to(train_args.device)
            actual = pathlib.Path(image_paths[i]).stem[:2] 
            imgIDs, scores, apks[i] = query(model_resnet, input_img, num, predictions, actual)
            img = Image.open(image_paths[index_image]).convert('LA')
            showResults(imgIDs)
            break

if __name__ == '__main__':
    main_func()
