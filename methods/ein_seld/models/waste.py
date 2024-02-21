  in_channels_sed = 4
  in_channels_doa = 7
        self.downsample_ratio = 2 ** 2   #sed_conv_block1

        if in_channels_sed < 4:
            # Pad the input channels with zeroes
            padding = nn.ZeroPad2d((0, 4 - in_channels_sed, 0, 0))
            self.sed[0] = nn.Sequential(padding, self.sed[0])

        if in_channels_doa < 7:
                   # Pad the input channels with zeroes
             padding = nn.ZeroPad2d((0, 7 - in_channels_doa, 0, 0))
             self.doa[0] = nn.Sequential(padding, self.doa[0])

        
        Transfer_Cnn14(classes_num =13, freeze_base = False),
                       nn.AvgPool2d(kernel_size=(2, 2)),

        )
        self.doa_conv_block1 = nn.Sequential(
           Transfer_Cnn14(in_channels_doa =7, in_channels_sed= 0, classes_num = 14, freeze_base =False),
            nn.AvgPool2d(kernel_size=(2, 2)), *list((Transfer_Cnn14.Cnn14.children)[:-1]), 

        

        self.sed = nn.Sequential(
              Transfer_Cnn14(in_channels=4, out_channels=64, classes_num = 14, freeze_base = False),
              nn.AvgPool2d(kernel_size=(2, 2),
        )
        # self.sed = nn.Sequential( 
         Transfer_Cnn14(in_channels_sed= 4,in_channels_doa=0,classes_num = 14, freeze_base = False ),
        *list(Transfer_Cnn14.named_children[1:]),  # Reuse the rest of the layers
        # #nn.Linear(cnn14_model.fc.in_features, num_classes_task1)
         )
       
        self.doa= nn.Sequential(
             Transfer_Cnn14(in_channels_doa= 7, out_channels=64, classes_num = 14, freeze_base = False),
                nn.AvgPool2d(kernel_size=(2, 2)
        )
        self.doa = nn.Sequential(
              Transfer_Cnn14(in_channels_doa=7, in_channels_sed= 0,classes_num = 14, freeze_base = False)
            #*list(Transfer_Cnn14.features[1:]),  # Reuse the rest of the layers
           nn.Linear(cnn14_model.fc.in_features, num_classes_task1)