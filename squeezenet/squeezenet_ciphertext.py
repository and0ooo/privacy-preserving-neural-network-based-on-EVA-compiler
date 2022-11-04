import numpy as np
import math
from PIL import Image
import pickle
import time
import psutil
import os
import argparse


from eva import EvaProgram, Input, Output
from eva.ckks import CKKSCompiler
from eva.seal import generate_keys
from eva import save, load


def my_padding(inputs, slots, inputs_channel, inputs_H, inputs_W, padding):

    masking = np.zeros(slots)
    inputs_padding = inputs * masking.tolist()

    masking[0:inputs_W] = 1
    for ic in range(0, inputs_channel):
        for row in range(0, inputs_H + 2 * padding):
            if row < padding or row >= inputs_H + padding:
                masking = np.roll(masking, inputs_W + 2 * padding)
                inputs = inputs >> (inputs_W + 2 * padding )
            else:
                masking = np.roll(masking, padding)
                inputs = inputs >> padding
                inputs_padding += inputs * masking.tolist()
                masking = np.roll(masking, inputs_W + padding)
                inputs = inputs >> padding
    return inputs_padding
        
# reference 1 (HEMET): https://arxiv.org/pdf/2106.00038.pdf
# reference 2: https://github.com/microsoft/EVA/blob/main/tests/large_programs.py
# reference 3: https://github.com/microsoft/EVA/blob/main/examples/image_processing.py
def my_conv_2d(inputs, slots, inputs_shape, filter_weights, stride, padding):

    inputs_channel, inputs_H, inputs_W = inputs_shape
    filter_num, filter_channel, filter_H, filter_W = filter_weights.shape

    # padding
    if padding > 0:
        assert slots >= inputs_channel * (inputs_H + 2 * padding) * (inputs_W + 2 * padding)
        inputs = my_padding(inputs, slots, inputs_channel, inputs_H, inputs_W, padding)
    
    # conv
    assert slots >= filter_num * math.floor((inputs_H + 2 * padding - filter_H) / stride + 1)  \
                               * math.floor((inputs_W + 2 * padding - filter_W) / stride + 1)
    width = inputs_W + 2 * padding
    height = inputs_H + 2 * padding
    masking = np.zeros(slots)
    inputs_convolved = inputs * masking.tolist()
    masking[0] = 1
    offset = 0
    for fn in range(0, filter_num):
        for c in range(0, filter_channel):
            for i in range(0, filter_H):
                for j in range(0, filter_W):
                    rotated = inputs << (c * width * height + i * width + j)
                    partial = rotated * filter_weights[fn,c,i,j]
                    if c == 0 and i == 0 and j == 0:
                        convolved = partial
                    else:
                        convolved += partial
        # masking and compression
        for r in range(0, inputs_H + 2 * padding - filter_H + 1, stride):
            for c in range(0, inputs_W + 2 * padding - filter_W + 1, stride):
                if r * width + c - offset >= 0:
                    inputs_convolved_tmp = convolved << (r * width + c - offset)
                else:
                    inputs_convolved_tmp = convolved >> (offset - r * width - c)
                masking_tmp = np.roll(masking, offset)
                offset += 1
                inputs_convolved += inputs_convolved_tmp * masking_tmp.tolist()

    h = math.floor((inputs_H + 2 * padding - filter_H) / stride + 1)
    w = math.floor((inputs_W + 2 * padding - filter_W) / stride + 1)
    output_shape = (filter_num, h, w)        
    return inputs_convolved, output_shape

def my_batch_norm_2d(inputs, slots, inputs_shape, mean, var, gamma, beta):

    inputs_channel, inputs_H, inputs_W = inputs_shape
    masking = np.zeros(slots)
    result = inputs * masking.tolist()
    masking[0:(inputs_H * inputs_W)] = 1

    for ic in range(0, inputs_channel):
        d = gamma[ic] / (var[ic] + 1e-5)**0.5
        e = beta[ic] - gamma[ic] * mean[ic] / (var[ic] + 1e-5)**0.5
        tmp_res = (d * inputs + e) * masking.tolist()
        result += tmp_res
        masking = np.roll(masking, inputs_H * inputs_W)
    return result


def my_quadratic_polynomial_activation(inputs, slots, inputs_shape, weight):

    result = weight[0] * inputs**2 + weight[1] * inputs + weight[2]
    return result

    

def my_avgpool_2d(inputs, slots, inputs_shape, filter_size, stride, padding):

    inputs_channel, inputs_H, inputs_W = inputs_shape

    # padding
    if padding > 0:
        assert slots >= inputs_channel * (inputs_H + 2 * padding) * (inputs_W + 2 * padding)
        inputs = my_padding(inputs, slots, inputs_channel, inputs_H, inputs_W, padding)

    # avgpool
    masking = np.zeros(slots)
    result = inputs * masking.tolist()
    width = inputs_W + 2 * padding
    height = inputs_H + 2 * padding
    masking[0] = 1
    offset = 0
    for i in range(0, filter_size):
        for j in range(0, filter_size):
            rotated = inputs << (i * width + j)
            if i == 0 and j == 0:
                convolved = rotated
            else:
                convolved += rotated
    convolved = convolved * (1 / filter_size**2)
    # masking and compression
    for ic in range(0, inputs_channel):
        for r in range(0, inputs_H + 2 * padding - filter_size + 1, stride):
            for c in range(0, inputs_W + 2 * padding - filter_size + 1, stride):
                if ic * width * height + r * width + c - offset >= 0:
                    inputs_convolved_tmp = convolved << (ic * width * height + r * width + c - offset)
                else:
                    inputs_convolved_tmp = convolved >> (offset - ic * width * height - r * width - c)
                masking_tmp = np.roll(masking, offset)
                offset += 1
                result += inputs_convolved_tmp * masking_tmp.tolist()
    
    h = math.floor((inputs_H + 2 * padding - filter_size) / stride + 1)
    w = math.floor((inputs_W + 2 * padding - filter_size) / stride + 1)
    output_shape = (inputs_channel, h, w) 
    return result, output_shape


def my_concatenate(slots, inputs1, inputs1_shape, inputs2, inputs2_shape):

    inputs1_channel, inputs1_H, inputs1_W = inputs1_shape
    inputs2_channel, inputs2_H, inputs2_W = inputs2_shape

    assert slots >= inputs1_channel * inputs1_H * inputs1_W + inputs2_channel * inputs2_H * inputs2_W
    assert inputs1_H == inputs2_H
    assert inputs1_W == inputs2_W
    concat_output_shape = (inputs1_channel + inputs2_channel, inputs1_H, inputs1_W)
    tmp_inputs2 = inputs2 >> (inputs1_channel * inputs1_H * inputs1_W)
    output = inputs1 + tmp_inputs2
    return output, concat_output_shape
    

def my_fire(inputs, slots, inputs_shape,
            conv1_weight, bn1_mean, bn1_var, bn1_gamma, bn1_beta, qpa1_weight,
            conv2_weight, bn2_mean, bn2_var, bn2_gamma, bn2_beta, qpa2_weight,
            conv3_weight, bn3_mean, bn3_var, bn3_gamma, bn3_beta, qpa3_weight):

    conv1_output, conv1_output_shape = my_conv_2d(inputs=inputs, 
                                                  slots=slots, 
                                                  inputs_shape=inputs_shape, 
                                                  filter_weights=conv1_weight, 
                                                  stride=1, 
                                                  padding=0)
    qpa1_output = my_quadratic_polynomial_activation(inputs=conv1_output, 
                           slots=slots, 
                           inputs_shape=conv1_output_shape, 
                           weight=qpa1_weight)                                              
    bn1_output = my_batch_norm_2d(inputs=qpa1_output, 
                                  slots=slots,
                                  inputs_shape=conv1_output_shape,
                                  mean=bn1_mean, 
                                  var=bn1_var, 
                                  gamma=bn1_gamma,
                                  beta=bn1_beta)
    

    conv2_output, conv2_output_shape = my_conv_2d(inputs=bn1_output, 
                                                  slots=slots,
                                                  inputs_shape=conv1_output_shape, 
                                                  filter_weights=conv2_weight,
                                                  stride=1, 
                                                  padding=0)
    qpa2_output = my_quadratic_polynomial_activation(inputs=conv2_output, 
                           slots=slots, 
                           inputs_shape=conv2_output_shape, 
                           weight=qpa2_weight)    
    bn2_output = my_batch_norm_2d(inputs=qpa2_output, 
                                slots=slots,
                                inputs_shape=conv2_output_shape,
                                mean=bn2_mean, 
                                var=bn2_var,
                                gamma=bn2_gamma,
                                beta=bn2_beta)


    conv3_output, conv3_output_shape = my_conv_2d(inputs=bn1_output, 
                                                  slots=slots,
                                                  inputs_shape=conv1_output_shape, 
                                                  filter_weights=conv3_weight,
                                                  stride=1, 
                                                  padding=1)
    qpa3_output = my_quadratic_polynomial_activation(inputs=conv3_output, 
                           slots=slots, 
                           inputs_shape=conv3_output_shape, 
                           weight=qpa3_weight) 
    bn3_output = my_batch_norm_2d(inputs=qpa3_output, 
                                  slots=slots,
                                  inputs_shape=conv3_output_shape,
                                  mean=bn3_mean, 
                                  var=bn3_var,
                                  gamma=bn3_gamma,
                                  beta=bn3_beta)


    concat_output, concat_output_shape = my_concatenate(slots=slots,
                                                        inputs1=bn2_output,
                                                        inputs1_shape=conv2_output_shape, 
                                                        inputs2=bn3_output, 
                                                        inputs2_shape=conv3_output_shape)
    
    output_shape = concat_output_shape
    return concat_output, output_shape


def my_sqznet(img, slots, image_shape, param):

    conv1_output, conv1_output_shape = my_conv_2d(
                        inputs=img,
                        slots=slots,
                        inputs_shape=image_shape,
                        filter_weights=param["conv1.weight"], 
                        stride=1, 
                        padding=1)    
    qpa1_output = my_quadratic_polynomial_activation(inputs=conv1_output,
                           slots=slots,
                           inputs_shape=conv1_output_shape,
                           weight=param["qpa1.weight"])    
    bn1_output = my_batch_norm_2d(inputs=qpa1_output, 
                                  slots=slots,
                                  inputs_shape=conv1_output_shape,
                                  mean=param["bn1.running_mean"], 
                                  var=param["bn1.running_var"],
                                  gamma=param["bn1.weight"],
                                  beta=param["bn1.bias"])
    avgpool1_output, avgpool1_output_shape = my_avgpool_2d(inputs=bn1_output,
                                                           slots=slots,
                                                           inputs_shape=conv1_output_shape,
                                                           filter_size=3,
                                                           stride=2,
                                                           padding=1)
    

    fire2_output, fire2_output_shape = my_fire(inputs=avgpool1_output, slots=slots, inputs_shape=avgpool1_output_shape,
                        conv1_weight=param["fire2.conv1.weight"], bn1_mean=param["fire2.bn1.running_mean"], bn1_var=param["fire2.bn1.running_var"],
                        bn1_gamma=param["fire2.bn1.weight"], bn1_beta=param["fire2.bn1.bias"], qpa1_weight=param["fire2.qpa1.weight"],
                        conv2_weight=param["fire2.conv2.weight"], bn2_mean=param["fire2.bn2.running_mean"], bn2_var=param["fire2.bn2.running_var"],
                        bn2_gamma=param["fire2.bn2.weight"], bn2_beta=param["fire2.bn2.bias"], qpa2_weight=param["fire2.qpa2.weight"],
                        conv3_weight=param["fire2.conv3.weight"], bn3_mean=param["fire2.bn3.running_mean"], bn3_var=param["fire2.bn3.running_var"],
                        bn3_gamma=param["fire2.bn3.weight"], bn3_beta=param["fire2.bn3.bias"], qpa3_weight=param["fire2.qpa3.weight"])
     

    fire3_output, fire3_output_shape = my_fire(inputs=fire2_output, slots=slots, inputs_shape=fire2_output_shape,
                        conv1_weight=param["fire3.conv1.weight"], bn1_mean=param["fire3.bn1.running_mean"], bn1_var=param["fire3.bn1.running_var"],
                        bn1_gamma=param["fire3.bn1.weight"], bn1_beta=param["fire3.bn1.bias"], qpa1_weight=param["fire3.qpa1.weight"],
                        conv2_weight=param["fire3.conv2.weight"], bn2_mean=param["fire3.bn2.running_mean"], bn2_var=param["fire3.bn2.running_var"],
                        bn2_gamma=param["fire3.bn2.weight"], bn2_beta=param["fire3.bn2.bias"], qpa2_weight=param["fire3.qpa2.weight"],
                        conv3_weight=param["fire3.conv3.weight"], bn3_mean=param["fire3.bn3.running_mean"], bn3_var=param["fire3.bn3.running_var"],
                        bn3_gamma=param["fire3.bn3.weight"], bn3_beta=param["fire3.bn3.bias"], qpa3_weight=param["fire3.qpa3.weight"])

    avgpool2_output, avgpool2_output_shape = my_avgpool_2d(inputs=fire3_output, 
                                                           slots=slots,
                                                           inputs_shape=fire3_output_shape,
                                                           filter_size=3, 
                                                           stride=2,
                                                           padding=1)
    

    conv2_output, conv2_output_shape = my_conv_2d(
                        inputs=avgpool2_output,
                        slots=slots,
                        inputs_shape=avgpool2_output_shape,
                        filter_weights=param["conv2.weight"], 
                        stride=1, 
                        padding=1)  
    qpa2_output = my_quadratic_polynomial_activation(inputs=conv2_output,
                           slots=slots,
                           inputs_shape=conv2_output_shape,
                           weight=param["qpa2.weight"])   
    bn2_output = my_batch_norm_2d(inputs=qpa2_output, 
                                  slots=slots,
                                  inputs_shape=conv2_output_shape,
                                  mean=param["bn2.running_mean"], 
                                  var=param["bn2.running_var"],
                                  gamma=param["bn2.weight"],
                                  beta=param["bn2.bias"])
          

    conv3_output, conv3_output_shape = my_conv_2d(
                        inputs=bn2_output,
                        slots=slots,
                        inputs_shape=conv2_output_shape,
                        filter_weights=param["conv3.weight"], 
                        stride=1, 
                        padding=1)   
    qpa3_output = my_quadratic_polynomial_activation(inputs=conv3_output,
                           slots=slots,
                           inputs_shape=conv3_output_shape,
                           weight=param["qpa3.weight"])   
    bn2_output = my_batch_norm_2d(inputs=qpa3_output, 
                                  slots=slots,
                                  inputs_shape=conv3_output_shape,
                                  mean=param["bn3.running_mean"], 
                                  var=param["bn3.running_var"],
                                  gamma=param["bn3.weight"],
                                  beta=param["bn3.bias"])
         

    conv4_output, conv4_output_shape = my_conv_2d(
                        inputs=bn2_output, 
                        slots=slots,
                        inputs_shape=conv3_output_shape,
                        filter_weights=param["conv4.weight"], 
                        stride=1, 
                        padding=0)   
    qpa4_output = my_quadratic_polynomial_activation(inputs=conv4_output,
                           slots=slots,
                           inputs_shape=conv4_output_shape,
                           weight=param["qpa4.weight"])    
    bn4_output = my_batch_norm_2d(inputs=qpa4_output, 
                                  slots=slots,
                                  inputs_shape=conv4_output_shape,
                                  mean=param["bn4.running_mean"], 
                                  var=param["bn4.running_var"],
                                  gamma=param["bn4.weight"],
                                  beta=param["bn4.bias"])
    

    avgpool3_output, avgpool3_output_shape = my_avgpool_2d(inputs=bn4_output,
                                                           slots=slots,
                                                           inputs_shape=conv4_output_shape,
                                                           filter_size=8,
                                                           stride=8,
                                                           padding=0)
    return avgpool3_output, avgpool3_output_shape


def my_store(param, name):
    print("--------------------storing " + name + " operations start--------------------")
    starttime = time.time()
    prog = EvaProgram('prog', vec_size=SLOTS)
    with prog:
        img = Input('img')
        avgpool3_output, avgpool3_output_shape = my_sqznet(img=img, slots=SLOTS, image_shape=IMAGE_SHAPE, param=param)
        Output('result', avgpool3_output)
    prog.set_input_scales(40)
    prog.set_output_ranges(40)
    endtime = time.time()
    print("store " + name + " operations finished, time consumed: " + str(round(endtime - starttime, 4)) + "s")
    return prog

def my_compile(prog, name):

    print("--------------------start compile " + name + "--------------------")
    compiler = CKKSCompiler({'security_level':'0', 'warn_vec_size':'false'})
    starttime = time.time()
    compiled, params, signature = compiler.compile(prog)
    endtime = time.time()
    print(name + " compile finished, time consumed: " + str(round(endtime - starttime, 4)) + "s")

    eva_prog[name] = compiled
    if args.save:
        save(compiled, COMPILED_PROG_SAVE_PATH + name + ".evacompiled")
        print("'" + name + ".evacompiled' saved")
    else:
        print("'" + name + ".evacompiled' will not be saved")

    eva_params[name] = params
    if args.save:
        save(params, PARAMS_SAVE_PATH + name + ".evaparams")
        print("'" + name + ".evaparams' saved")
    else:
        print("'" + name + ".evaparams' will not be saved")

    eva_signature[name] = signature
    if args.save:
        save(signature, SIGNATURE_SAVE_PATH + name + ".evasignature")
        print("'"+ name +".evasignature' saved")
    else:
        print("'"+ name +".evasignature' will not be saved")
    check_memory()

def my_generate_keys(name):

    print("--------------------" + name + " start generate keys--------------------")
    starttime = time.time()
    public_ctx, secret_ctx = generate_keys(eva_params[name])
    endtime = time.time()
    print(name + " generate keys finished, time consumed: " + str(round(endtime - starttime, 4)) + "s")

    eva_public_ctx[name] = public_ctx
    if args.save:
        save(public_ctx, PUBLIC_CTX_SAVE_PATH + name + ".sealpublic")
        print("'" + name + ".sealpublic' saved")
    else:
        print("'" + name + ".sealpublic' will not be saved")

    eva_secret_ctx[name] = secret_ctx
    if args.save:
        save(secret_ctx, SECRET_CTX_SAVE_PATH + name + ".sealsecret")
        print("'" + name + ".sealsecret' saved")
    else:
        print("'" + name + ".sealsecret' will not be saved")
    check_memory()

def my_enc(inputs, signature, public_ctx, name):

    print(name + " start encrypt----------------------------------------")
    starttime = time.time()
    enc_inputs = public_ctx.encrypt(inputs, signature)
    endtime = time.time()
    print(name + " encrypt finished, time consumed: " + str(round(endtime - starttime, 4)) + "s")
    check_memory()
    return enc_inputs

def my_exec(compiled, enc_inputs, public_ctx, name):

    print("--------------------" + name + " start execute--------------------")
    starttime = time.time()
    enc_outputs = public_ctx.execute(compiled, enc_inputs)
    endtime = time.time()
    print(name + " execute finished, time consumed: " + str(round(endtime - starttime, 4)) + "s")
    check_memory()
    return enc_outputs

def my_dec(enc_outputs, signature, secret_ctx, name):

    print("----------------------------------------" + name + " start decrypt")
    starttime = time.time()
    outputs = secret_ctx.decrypt(enc_outputs, signature)
    endtime = time.time()
    print(name + " decrypt finished, time consumed: " + str(round(endtime - starttime, 4)) + "s")
    check_memory()
    return outputs

def check_memory():

    mem = psutil.virtual_memory()
    print("memory check | total memory: {} GB; used memory: {} GB; free memory: {} GB".format(
            round(float(mem.total) / 1024 / 1024 / 1024, 4), 
            round(float(mem.used) / 1024 / 1024 / 1024, 4), 
            round(float(mem.free) / 1024 / 1024 / 1024, 4)
        )
    )

def load_after_check_exists(check, name):
    
    print("--------------------checking " + name + " start--------------------")
    check[name] = {}

    if os.path.exists(COMPILED_PROG_SAVE_PATH + name + ".evacompiled"):
        eva_prog[name] = load(COMPILED_PROG_SAVE_PATH + name + ".evacompiled")
        print("'" + name + ".evacompiled'" + " loaded")
        check[name]["compiled_exsits"] = True
    else:
        print("'" + name + ".evacompiled'" + " does not exist")
        check[name]["compiled_exsits"] = False

    if os.path.exists(PARAMS_SAVE_PATH + name + ".evaparams"):
        eva_params[name] = load(PARAMS_SAVE_PATH + name + ".evaparams")
        print("'" + name + ".evaparams'" + " loaded")
        check[name]["params_exsits"] = True
    else:
        print("'" + name + ".evaparams'" + " does not exist")
        check[name]["params_exsits"] = False

    if os.path.exists(SIGNATURE_SAVE_PATH + name + ".evasignature"):
        eva_signature[name] = load(SIGNATURE_SAVE_PATH + name + ".evasignature")
        print("'" + name + ".evasignature'" + " loaded")
        check[name]["signature_exsits"] = True
    else:
        print("'" + name + ".evasignature'" + " does not exist")
        check[name]["signature_exsits"] = False
    
    if os.path.exists(PUBLIC_CTX_SAVE_PATH + name + ".sealpublic"):
        eva_public_ctx[name] = load(PUBLIC_CTX_SAVE_PATH + name + ".sealpublic")
        print("'" + name + ".sealpublic'" + " loaded")
        check[name]["public_ctx_exsits"] = True
    else:
        print("'" + name + ".sealpublic'" + " does not exist")
        check[name]["public_ctx_exsits"] = False
    
    if os.path.exists(SECRET_CTX_SAVE_PATH + name + ".sealsecret"):
        eva_secret_ctx[name] = load(SECRET_CTX_SAVE_PATH + name + ".sealsecret")
        print("'" + name + ".sealsecret'" + " loaded")
        check[name]["secret_ctx_exsits"] = True
    else:
        print("'" + name + ".sealsecret'" + " does not exist")
        check[name]["secret_ctx_exsits"] = False


parser = argparse.ArgumentParser("my privacy preserving SqueezeNet")
parser.add_argument("-s", '--save', action='store_true', default=False, help='Save to files. Default is false.')
parser.add_argument("-r", '--read', action='store_true', default=True, help='Read from files. Default is true.')
parser.add_argument("-p", '--path', type=str, default="/root/autodl-tmp/", help="If 'save' is true, 'path' defines the saving path.")
parser.add_argument("-i", '--img', type=str, default="0_10.jpg", help="Path of img.")
parser.add_argument("-a", '--param', type=str, default="squeezenet_param_4096_7468.pkl", help="Path of params.")
args = parser.parse_args()

SLOTS = 4096
print("slots: " + str(SLOTS))
IMAGE_SHAPE = (3, 32, 32)
IMAGE_PAD = SLOTS - IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * IMAGE_SHAPE[2]
CIFAR10_TEST_MEAN = (0.491399689874, 0.482158419622, 0.446530924224)
CIFAR10_TEST_STD = (0.247032237587, 0.243485133253, 0.261587846975)


PATH = args.path
COMPILED_PROG_SAVE_PATH = PATH + "prog/"
PARAMS_SAVE_PATH = PATH + "params/"
SIGNATURE_SAVE_PATH = PATH + "sig/"
PUBLIC_CTX_SAVE_PATH = PATH + "pc/"
SECRET_CTX_SAVE_PATH = PATH + "sc/"
OUTPUT_PATH = PATH + "output/"

if args.save:
    if not os.path.exists(COMPILED_PROG_SAVE_PATH):
        os.mkdir(COMPILED_PROG_SAVE_PATH)
    if not os.path.exists(PARAMS_SAVE_PATH):
        os.mkdir(PARAMS_SAVE_PATH)
    if not os.path.exists(SIGNATURE_SAVE_PATH):
        os.mkdir(SIGNATURE_SAVE_PATH)
    if not os.path.exists(PUBLIC_CTX_SAVE_PATH):
        os.mkdir(PUBLIC_CTX_SAVE_PATH)
    if not os.path.exists(SECRET_CTX_SAVE_PATH):
        os.mkdir(SECRET_CTX_SAVE_PATH)
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

eva_prog = {}
eva_params = {}
eva_signature = {}
eva_public_ctx = {}
eva_secret_ctx = {}

if __name__ == '__main__':

    # read params
    with open(args.param, 'rb') as f:
        param = pickle.load(f)

    # read img
    img_path = args.img
    img = Image.open(img_path)
    # preprocessing
    img = np.array(img)
    img_R = img[:,:,0]
    img_G = img[:,:,1]
    img_B = img[:,:,2]
    img = np.array([img_R,img_G,img_B])
    img = img / 255
    img[0] = (img[0] - CIFAR10_TEST_MEAN[0]) / CIFAR10_TEST_STD[0]
    img[1] = (img[1] - CIFAR10_TEST_MEAN[1]) / CIFAR10_TEST_STD[1]
    img[2] = (img[2] - CIFAR10_TEST_MEAN[2]) / CIFAR10_TEST_STD[2]
    img = img.reshape(-1)
    img = np.append(img, np.array([0 for _ in range(IMAGE_PAD)]))
    inputs = {'img': img}

    name = "sqznet"

    # check existence
    if args.read:
        check = {}
        load_after_check_exists(check, name)
        print(name + " checking finished")
    else:
        print("Compiled program, params, sig, public_ctx and secret_ctx will not be read from files.")

    # compile
    if not args.read:
        prog = my_store(param, name)
        my_compile(prog, name)
    elif args.read and not (check[name]["compiled_exsits"] and check[name]["params_exsits"] and check[name]["signature_exsits"]):
        prog = my_store(param, name)
        my_compile(prog, name)
    else:
        print("Compiled program, params and sig have been read from files.")
    
    # generate keys
    if not args.read:
        my_generate_keys(name)
    elif args.read and not (check[name]["public_ctx_exsits"] and check[name]["secret_ctx_exsits"]):
        my_generate_keys(name)
    else:
        print("Public_ctx and secret_ctx have been read from files.")

    # encrypt + execute + decrypt
    enc_inputs = my_enc(inputs, eva_signature[name], eva_public_ctx[name], name)
    enc_outputs = my_exec(eva_prog[name], enc_inputs, eva_public_ctx[name], name)
    outputs = my_dec(enc_outputs, eva_signature[name], eva_secret_ctx[name], name)

    # output
    outputs = outputs["result"]
    print("--------------------start output--------------------")
    out = ""
    columns = 4
    for i in range(len(outputs)):
        if i != 0 and i % columns == 0:
            out += "\n"
        out += str(outputs[i]) + " "
    with open(OUTPUT_PATH + "outputs.txt",'w')as file:
        file.write(out)
    print("output finished")
    check_memory()

    # prediction
    print("output:" + str(outputs[0:15]))
    res = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"] 
    pred = outputs.argmax()
    print(img_path + " - prediction class: " + str(pred) + " (" + res[pred] + ")")


