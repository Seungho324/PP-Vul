from seal import *

from pp_vul.utils.functional import inner_sum
from pp_vul.utils.context import Context
from pp_vul.utils.structure import Output, Cuboid, Rectangle

# def conv2d_layer_converter_(context: Context, In: Output, Img: Cuboid, layer, data_size):
#     """
#     This function calculates the 2D convolution operation of the input data.
    
#     Args:
#         - context: Context that has all the necessary keys
#             - evaluator: CKKS Evaluator in the SEAL-Python library
#             - encoder: CKKS Encoder in the SEAL-Python library
#             - galois_key: CKKS galois key in the SEAL-Python library
#             - relin_keys: Re-linearization key of CKKS scheme in the SEAL-Python library
#         - In: This is containing the information below
#             - ciphertexts: Input ciphertexts list
#             - size: Size of input data that is removed the invalid values
#             - interval: Interval value between valid data before Conv2d layer
#             - const: Value to be multiplied by ciphertext before layer
#         - Img: Width (and height) of used image data
#         - layer: Convolutional 2D layer that is containing the information below
#             - in_channels: Number of input channels
#             - out_channels: Number of output channels
#             - weight: Kernel weight (shape: CH_out * CH_in * K.h * K.w)
#             - bias: Bias value (shape: CH_out)
#             - padding: Padding size
#             - stride: Stride value
#         - data_size: Maximum data size from the total layers

#     Returns:
#         - Out: This is containing the information below
#             - ciphertexts: Output ciphertexts list
#             - size: Size of output data that is removed the invalid  values
#             - interval: Interval value between valid data after Conv2d layer
#             - const: Value to be multiplied by C_out after Conv2d layer (= 1)
#     """ 
#     # kernel size = 3, out channel = 32
#     # rot: 768
#     # mul: 24576
    
#     CH_in   = layer.in_channels
#     CH_out  = layer.out_channels
#     K       = Rectangle(layer.kernel_size[0], layer.kernel_size[1])
#     S       = Rectangle(layer.stride[0], layer.stride[1])
#     P       = Rectangle(layer.padding[0], layer.padding[1])
    
#     Out = Output(
#         ciphertexts = [], 
#         size = Cuboid(
#             length = CH_out,
#             height = (In.size.h + 2 * P.h - K.h) // S.h + 1,
#             width = (In.size.w + 2 * P.w - K.w) // S.w + 1
#         ),
#         interval = Rectangle(In.interval.h * S.h, In.interval.w * S.w), 
#         const = 1
#     )
    
#     C_rot = []

#     for i in range(CH_in):
#         C_rot.append([])
#         for p in range(K.h):
#             C_rot[i].append([])
#             for q in range(K.w):
#                 ciphertext = context.evaluator.rotate_vector(
#                     In.ciphertexts[i],
#                     In.interval.h * Img.w * p + In.interval.w * q, 
#                     context.galois_key
#                 )
#                 C_rot[i][p].append(ciphertext) 
    
#     for o in range(CH_out):
#         C_outs = []
#         for i in range(CH_in):
#             for p in range(K.h):
#                 for q in range(K.w):
#                     """Vector of kernel"""
#                     V_ker = [layer.weight.detach().tolist()[o][i][p][q] * In.const] + [0] * (Out.interval.w - 1)
#                     V_ker = V_ker * Out.size.w + [0] * (Img.w * Out.interval.h - Out.size.w * Out.interval.w)
#                     V_ker = V_ker * Out.size.h + [0] * (data_size - Img.w * Out.interval.h * Out.size.h)
#                     V_ker = V_ker * (context.number_of_slots // data_size)

#                     Plaintext_ker = context.encoder.encode(V_ker, context.scale)
#                     context.evaluator.mod_switch_to_inplace(Plaintext_ker, C_rot[i][p][q].parms_id())

#                     """
#                     This try-catch part is handling exceptions for errors that occur when multiplying the vector of 0.
#                     """
#                     try:
#                         ciphertext = context.evaluator.multiply_plain(C_rot[i][p][q], Plaintext_ker)
#                         context.evaluator.relinearize_inplace(ciphertext, context.relin_keys)
#                         context.evaluator.rescale_to_next_inplace(ciphertext)

#                         C_outs.append(ciphertext)
#                     except RuntimeError as e:
#                         print("Warning: An error occurred, but it's being ignored:", str(e))
        
#         ciphertext = context.evaluator.add_many(C_outs)

#         """Vector of bias"""            
#         V_bias = [layer.bias.detach().tolist()[o]] + [0] * (Out.interval.w - 1)  
#         V_bias = V_bias * Out.size.w + [0] * (Img.w * Out.interval.h - Out.size.w * Out.interval.w)
#         V_bias = V_bias * Out.size.h + [0] * (data_size - Img.w * Out.interval.h * Out.size.h)
#         V_bias = V_bias * (context.number_of_slots // data_size) 

#         Plaintext_bias = context.encoder.encode(V_bias, ciphertext.scale())
#         context.evaluator.mod_switch_to_inplace(Plaintext_bias, ciphertext.parms_id())
#         ciphertext = context.evaluator.add_plain(ciphertext, Plaintext_bias)
#         Out.ciphertexts.append(ciphertext)
        
#     return Out

def conv2d_layer_converter(context: Context, In: Output, Img: Cuboid, layer, data_size):
    CH_in   = layer.in_channels
    CH_out  = layer.out_channels
    K       = Rectangle(layer.kernel_size[0], layer.kernel_size[1])
    S       = Rectangle(layer.stride[0], layer.stride[1])
    P       = Rectangle(layer.padding[0], layer.padding[1])

    Out = Output(
        ciphertexts = [], 
        size = Cuboid(
            length = CH_out,
            height = (In.size.h + 2 * P.h - K.h) // S.h + 1,
            width = (In.size.w + 2 * P.w - K.w) // S.w + 1
        ),
        interval = Rectangle(In.interval.h * S.h, In.interval.w * S.w), 
        const = 1
    )
    
    C_rot = []
    
    rot_num = 0
    for p in range(K.h):            
        ciphertext = context.evaluator.rotate_vector(
            In.ciphertexts[0],
            K.w * p,
            context.galois_key
        )
        rot_num = rot_num + 1
        C_rot.append(ciphertext)     
    
    for o in range(CH_out):
        C_outs = []
        for p in range(K.h):
            """Vector of kernel"""
            V_ker = layer.weight.detach().tolist()[o][0][p]
            V_ker = V_ker * Out.size.h
            Plaintext_ker = context.encoder.encode(V_ker, context.scale)   
                
            context.evaluator.mod_switch_to_inplace(Plaintext_ker, C_rot[p].parms_id())    
            try:
                ciphertext = context.evaluator.multiply_plain(C_rot[p], Plaintext_ker)
                context.evaluator.relinearize_inplace(ciphertext, context.relin_keys)
                context.evaluator.rescale_to_next_inplace(ciphertext)
                C_outs.append(ciphertext)
            except RuntimeError as e:
                print("Warning: An error occurred, but it's being ignored:", str(e))

        ciphertext = context.evaluator.add_many(C_outs)
        ciphertext = inner_sum(context, ciphertext, K.w)

        """Vector of bias"""
        V_bias = [layer.bias.detach().tolist()[o]] + [0] * (Img.w - 1)
        V_bias = V_bias * Out.size.h

        Plaintext_bias = context.encoder.encode(V_bias, ciphertext.scale())
        context.evaluator.mod_switch_to_inplace(Plaintext_bias, ciphertext.parms_id())
        ciphertext = context.evaluator.add_plain(ciphertext, Plaintext_bias)                        
        Out.ciphertexts.append(ciphertext)

    return Out

# def inner_sum(context: Context, ciphertext_add, data_size, batch_size: int=1):
#     num_rot = math.ceil(data_size / batch_size)
#     ciphertext_temp = -1

#     while num_rot > 1:
#         if num_rot % 2 == 1:
#             ciphertext_rot = context.evaluator.rotate_vector(
#                 ciphertext_add,
#                 (num_rot - 1) * batch_size,
#                 context.galois_key
#             )
            
#             if ciphertext_temp == -1:
#                 ciphertext_temp = ciphertext_rot
#             else:
#                 ciphertext_temp = context.evaluator.add(ciphertext_temp, ciphertext_rot)

#         num_rot = num_rot // 2
#         ciphertext_rot = context.evaluator.rotate_vector(
#             ciphertext_add,
#             num_rot * batch_size, 
#             context.galois_key
#         )
#         ciphertext_add = context.evaluator.add(ciphertext_add, ciphertext_rot)
#     ciphertext_add = context.evaluator.add(ciphertext_add, ciphertext_temp)
    
#     return ciphertext_add

# def gather_ciphertext(context: Context, In: list, data_num):
#     C_add = []
    
#     for i in range(data_num):
#         V_clean = [1] + [0] * 16383
#         Plaintext_clean = context.encoder.encode(V_clean, context.scale)
#         context.evaluator.mod_switch_to_inplace(Plaintext_clean, In[i].parms_id())
#         ciphertext = context.evaluator.multiply_plain(In[i], Plaintext_clean)
#         context.evaluator.rescale_to_next_inplace(ciphertext)
        
#         ciphertext_rot = context.evaluator.rotate_vector(
#             ciphertext,
#             -i, 
#             context.galois_key
#         )
#         C_add.append(ciphertext_rot)
        
#     ciphertext = context.evaluator.add_many(C_add)
    
#     return ciphertext
        
def conv2d_layer_converter_depthwise(context: Context, In: Output, Img: Cuboid, layer, data_size):
    CH_in   = layer.in_channels
    CH_out  = layer.out_channels
    if CH_in != CH_out:
        raise ValueError("Input channel is not equal to output channel.")
    K       = Rectangle(layer.kernel_size[0], layer.kernel_size[1])
    S       = Rectangle(layer.stride[0], layer.stride[1])
    P       = Rectangle(layer.padding[0], layer.padding[1])

    Out = Output(
        ciphertexts = [], 
        size = Cuboid(
            length = CH_out,
            height = (In.size.h + 2 * P.h - K.h) // S.h + 1,
            width = (In.size.w + 2 * P.w - K.w) // S.w + 1
        ),
        interval = Rectangle(In.interval.h * S.h, In.interval.w * S.w), 
        const = 1
    )

    for o in range(CH_out):
        V_ker = []
        for q in range(K.h):
            """Vector of kernel"""
            V_ker = V_ker + [layer.weight.detach().tolist()[o][0][q][0]] + [0] * (Img.w - 1)       
        
        Plaintext_ker = context.encoder.encode(V_ker, context.scale)
        context.evaluator.mod_switch_to_inplace(Plaintext_ker, In.ciphertexts[o].parms_id())
        try:
            ciphertext = context.evaluator.multiply_plain(In.ciphertexts[o], Plaintext_ker)
            context.evaluator.relinearize_inplace(ciphertext, context.relin_keys)
            context.evaluator.rescale_to_next_inplace(ciphertext)
                        
        except RuntimeError as e:
            print("Warning: An error occurred, but it's being ignored:", str(e))
        
        ciphertext = inner_sum(context, ciphertext, K.h * Img.w, Img.w)

        """Vector of bias"""
        V_bias = [layer.bias.detach().tolist()[o]]
        
        Plaintext_bias = context.encoder.encode(V_bias, ciphertext.scale())        
        context.evaluator.mod_switch_to_inplace(Plaintext_bias, ciphertext.parms_id())
        ciphertext = context.evaluator.add_plain(ciphertext, Plaintext_bias)
        Out.ciphertexts.append(ciphertext)          
                
    return Out