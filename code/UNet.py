from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D,UpSampling3D, Dropout,concatenate, BatchNormalization, Activation, Add, ReLU
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def initialization():
    # load tf
    print(tf.__version__)
    K.set_floatx('float32')

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
     
    devices_names = [d.name.split('e:')[1] for d in gpus]
    strategy = tf.device(devices_names[0])#tf.distribute.MirroredStrategy(devices=devices_names[:],cross_device_ops=tf.distribute.ReductionToOneDevice())
    return strategy

def dice_coef(y_true, y_pred, smooth=0.1):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
  return -dice_coef(y_true, y_pred)

def UNet(strategy,input_shape,initial_learning_rate=1e-5,classes = 1, start_filter = 32,pool_size=[2,2,2], conv_size=[3,3,3],drop_out=0.2,name ='UNet',regression=False,val=1500):
    
    
    
    with strategy.scope():
        
        print('Compiling UNET model')
        inputs = Input(input_shape, name='model_input')
        conv1 = Conv3D(start_filter, conv_size, activation='relu', padding='same', name='conv_1_1')(inputs)
        norm1 = BatchNormalization(axis=4, name='norm_1_1')(conv1)
        conv1 = Conv3D(2*start_filter, conv_size, activation='relu', padding='same', name='conv_1_2')(norm1)
        norm1 = BatchNormalization(axis=4, name='norm_1_2')(conv1)
        pool1 = MaxPooling3D(pool_size=pool_size, name='pool_1')(norm1)
          
        conv2 = Conv3D(2*start_filter, conv_size, activation='relu', padding='same', name='conv_2_1')(pool1)
        norm2 = BatchNormalization(axis=4, name='norm_2_1')(conv2)
        conv2 = Conv3D(4*start_filter, conv_size, activation='relu', padding='same', name='conv_2_2')(norm2)
        norm2 = BatchNormalization(axis=4, name='norm_2_2')(conv2)
        pool2 = MaxPooling3D(pool_size=pool_size, name='pool_2')(norm2)
          
        conv3 = Conv3D(4*start_filter, conv_size, activation='relu', padding='same', name='conv_3_1')(pool2)
        norm3 = BatchNormalization(axis=4, name='norm_3_1')(conv3)
        conv3 = Conv3D(8*start_filter, conv_size, activation='relu', padding='same', name='conv_3_2')(norm3)
        norm3 = BatchNormalization(axis=4, name='norm_3_2')(conv3)
        pool3 = MaxPooling3D(pool_size=pool_size, name='pool_3')(norm3)
          
        conv4 = Conv3D(8*start_filter, conv_size, activation='relu', padding='same', name='conv_4_1')(pool3)
        norm4 = BatchNormalization(axis=4, name='norm_4_1')(conv4)
        conv4 = Conv3D(16*start_filter, conv_size, activation='relu', padding='same', name='conv_4_2')(norm4)
        norm4 = BatchNormalization(axis=4, name='norm_4_2')(conv4)
        pool4 = MaxPooling3D(pool_size=pool_size, name='pool_4')(norm4)
          
        conv5 = Conv3D(8*start_filter, conv_size, activation='relu', padding='same', name='conv_5_1')(pool4)
        norm5 = BatchNormalization(axis=4, name='norm_5_1')(conv5)
        conv5 = Conv3D(16*start_filter, conv_size, activation='relu', padding='same', name='conv_5_2')(norm5)
        norm5 = BatchNormalization(axis=4, name='norm_5_2')(conv5)
          
        up6 = UpSampling3D(size=pool_size, name='up_6')(norm5)
        up6 = concatenate([up6, norm4], axis=4, name='conc_6')
        drop6 = Dropout(rate=drop_out, name='drop_6')(up6)
        conv6 = Conv3D(8*start_filter, conv_size, activation='relu', padding='same', name='conv_6_1')(drop6)
        conv6 = Conv3D(8*start_filter, conv_size, activation='relu', padding='same', name='conv_6_2')(conv6)
          
        up7 = UpSampling3D(size=pool_size, name='up_7')(conv6)
        up7 = concatenate([up7, norm3], axis=4, name='conc_7')
        drop7 = Dropout(rate=drop_out, name='drop_7')(up7)
        conv7 = Conv3D(4*start_filter, conv_size, activation='relu', padding='same', name='conv_7_1')(drop7)
        conv7 = Conv3D(4*start_filter, conv_size, activation='relu', padding='same', name='conv_7_2')(conv7)
          
        up8 = UpSampling3D(size=pool_size, name='up_8')(conv7)
        up8 = concatenate([up8, norm2], axis=4, name='conc_8')
        drop8 = Dropout(rate=drop_out, name='drop_8')(up8)
        conv8 = Conv3D(2*start_filter, conv_size, activation='relu', padding='same', name='conv_8_1')(drop8)
        conv8 = Conv3D(2*start_filter, conv_size, activation='relu', padding='same', name='conv_8_2')(conv8)
          
        up9 = UpSampling3D(size=pool_size, name='up_9')(conv8)
        up9 = concatenate([up9, norm1], axis=4, name='conc_9')
        drop9 = Dropout(rate=drop_out, name='drop_9')(up9)
        conv9 = Conv3D(2*start_filter, conv_size, activation='relu', padding='same', name='conv_9_1')(drop9)
        conv9 = Conv3D(2*start_filter, conv_size, activation='relu', padding='same', name='conv_9_2')(conv9)
          
        
        
        

        if classes ==1:
            
            if regression:
                
                lossName = 'log_cosh'
                # lossName = 'mean_absolute_percentage_error'
                #loss = weighted_abs_error
                #lossName = 'weighted_abs_error'
                
                
                conv9 = resBlock(conv9,2*start_filter,conv_size,activate=False)
                conv10 = Conv3D(1, (1, 1, 1), name='conv_10')(conv9)
                act = ReLU(max_value=val)(conv10)
                
                print('Compiling single output model')
                model = Model(inputs=inputs, outputs=act,name='RNET')
    
                model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss=lossName,metrics=[lossName])
                monitor = 'val_'+lossName
            else:
                conv9 = resBlock(conv9,2*start_filter,conv_size,activate=True)
                conv10 = Conv3D(1, (1, 1, 1), name='conv_10')(conv9)
                act = Activation('sigmoid', name='act')(conv10)
                
                print('Compiling single output model')
                model = Model(inputs=inputs, outputs=act,name='RNET')
    
                model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss=dice_coef_loss,metrics=[dice_coef])
                monitor = 'val_dice_coef'
                
        else:
            print('Compiling multiple output model')
            conv9 = resBlock(conv9,2*start_filter,conv_size,activate=True)
            conv10 = Conv3D(classes, (1, 1, 1), name='conv_10')(conv9)
            act = Activation('softmax', name='act')(conv10)
            
            multi_class_dice_coef_loss = init_multi_class_dice_coef_loss(classes =classes)
            multi_class_dice_coef = init_multi_class_dice_coef(classes =classes)
            
            model = Model(inputs=inputs, outputs=act,name=name)
            model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss=multi_class_dice_coef_loss,metrics=multi_class_dice_coef ) 

            monitor = 'val_multi_class_dice_coef'
        return model,monitor


