import torch
import torch.nn as nn



# Siamese model class
class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.fc_out = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())     
        self.fc_out_image = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid()) 
        self.fc_out_rock = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid()) 
        self.fc_fusion = nn.Sequential(nn.Linear(1024, 32), nn.ReLU()) 
        self.fc_fusion2 = nn.Sequential(nn.Linear(32, 16), nn.ReLU()) 
        self.fc_fusion3 = nn.Sequential(nn.Linear(16, 2), nn.ReLU()) 


    # Based on the figure 1 of the paper
    # x1 : output(h1) from the blue part of network
    # x2 : output(h2) from the blue part of network
    # x3 : output(h3) from the gray part of network
    # x4 : output(h4) from the gray part of network
    def forward(self, x1, x2, x3, x4):
        out1 = x1 
        out2 = x2  
        out3 = x3  
        out4 = x4 
        
        dis = torch.abs(out1 - out2)      
        dis_rock = torch.abs(out3 - out4)

        m = nn.AdaptiveAvgPool2d((1, 1))
        out_img = m(dis)
        out_img_flat = torch.flatten(out_img, 1)  # out_img_flat = w1
        out_img = self.fc_out_image(out_img_flat) # out_img = FC(w1) 

        out_rock = m(dis_rock)
        out_rock_flat = torch.flatten(out_rock, 1) # out_rock_flat = w2
        out_rock = self.fc_out_rock(out_rock_flat) # out_rock = FC(w2)

        # Initial fusion
        # fused_out = (out_img_flat + out_rock_flat)/2 
        fused_out = torch.cat((out_img_flat, out_rock_flat), dim=1)

        # Gating 
        fused_out = self.fc_fusion(fused_out)
        fused_out = self.fc_fusion2(fused_out)
        fused_scalars = self.fc_fusion3(fused_out)
        m = nn.Softmax()
        fused_scalars = m(fused_scalars) # fused_scalars =[s1,s2]

        out_img_flat = out_img_flat * fused_scalars[:, 0].view(-1, 1) # scaled out_img_flat = w1'
        out_rock_flat = out_rock_flat * fused_scalars[:, 1].view(-1, 1)  # scaled out_rock_flat = w2'
        fused_out = (out_img_flat + out_rock_flat)/2 # w=0.5*(w1'+w2')

        out = self.fc_out(fused_out) # out=FC(w)

        return out, out_img, out_rock



# Regression module class
class Regression_rock(nn.Module):

    def __init__(self):
        super(Regression_rock, self).__init__()
        self.fc = nn.Sequential(nn.Linear(512,64), nn.ReLU())
        self.out_por = nn.Sequential(nn.Linear(64,1))
        self.out_perm = nn.Sequential(nn.Linear(64,1))

    # x is either h3 or h4, used in main code
    def forward(self, x):
        m = nn.AdaptiveAvgPool2d((1,1))
        x = m(x)
        fc_out = self.fc(x.view(x.size(0), -1))
        out_por = self.out_por(fc_out)
        out_perm = self.out_perm(fc_out)
        out = torch.cat([out_por, out_perm], dim=1)

        return out



# for test
if __name__ == '__main__':
    net = Siamese()
    print(net)

