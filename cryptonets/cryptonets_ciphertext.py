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
    offset = 0
    for ic in range(0, inputs_channel):
        for row in range(0, inputs_H + 2 * padding):
            if row < padding or row >= inputs_H + padding:
                masking = np.roll(masking, inputs_W + 2 * padding)
                offset += inputs_W + 2 * padding
            else:
                masking = np.roll(masking, padding)
                offset += padding
                inputs_padding += (inputs >> offset) * masking.tolist()
                masking = np.roll(masking, inputs_W + padding)
                offset += padding
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

def my_quadratic_polynomial_activation(inputs, slots, weight):

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

# reference: https://github.com/microsoft/EVA/issues/6#issuecomment-761295103
def my_sum(x, slots):
    i = slots // 2
    while i >= 1:
        y = x << i
        x = x + y
        i >>= 1    
    masking = np.zeros(slots)
    masking[0] = 1
    return x * masking.tolist()


def my_linear(inputs, slots, weights):
    rows, columns = weights.shape
    for r in range(rows):
        mul = np.append(weights[r], np.array([0 for _ in range(slots - columns)]))
        if r == 0:
            result = my_sum(inputs * mul.tolist(), slots)
        else:
            result += my_sum(inputs * mul.tolist(), slots) >> r
    return result

def my_cryptonets(img, slots, image_shape, param):

    conv1_output, conv1_output_shape = my_conv_2d(
                        inputs=img,
                        slots=slots,
                        inputs_shape=image_shape,
                        filter_weights=param["conv1.weight"], 
                        stride=2, 
                        padding=1)
    
    qpa1_output = my_quadratic_polynomial_activation(inputs=conv1_output,
                           slots=slots,
                           weight=param["qpa1.weight"])
    avgpool1_output, avgpool1_output_shape = my_avgpool_2d(inputs=qpa1_output,
                                                           slots=slots,
                                                           inputs_shape=conv1_output_shape,
                                                           filter_size=3,
                                                           stride=1,
                                                           padding=1)
    
    conv2_output, conv2_output_shape = my_conv_2d(
                        inputs=avgpool1_output,
                        slots=slots,
                        inputs_shape=avgpool1_output_shape,
                        filter_weights=param["conv2.weight"], 
                        stride=2, 
                        padding=0)
    avgpool2_output, avgpool2_output_shape = my_avgpool_2d(inputs=conv2_output,
                                                           slots=slots,
                                                           inputs_shape=conv2_output_shape,
                                                           filter_size=3,
                                                           stride=1,
                                                           padding=1)
    
    linear3_output = my_linear(inputs=avgpool2_output,
                              slots=slots, 
                              weights=param["fc3.weight"])
    
    return linear3_output

def my_store(param, name):
    print("--------------------storing " + name + " operations start--------------------")
    starttime = time.time()
    prog = EvaProgram('prog', vec_size=SLOTS)
    with prog:
        img = Input('img')
        linear4_output = my_cryptonets(img=img, slots=SLOTS, image_shape=IMAGE_SHAPE, param=param)
        Output('result', linear4_output)
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


parser = argparse.ArgumentParser("my CryptoNets")
parser.add_argument("-s", '--save', action='store_true', default=False, help='Save to files. Default is false.')
parser.add_argument("-r", '--read', action='store_true', default=True, help='Read from files. Default is true.')
parser.add_argument("-p", '--path', type=str, default="", help="If 'save' is true, 'path' defines the saving path.")
parser.add_argument("-i", '--img', type=str, default="0.jpg", help="Path of img.")
parser.add_argument("-a", '--param', type=str, default="cryptonets_param.pkl", help="Path of params.")
args = parser.parse_args()

SLOTS = 1024
print("slots: " + str(SLOTS))
IMAGE_SHAPE = (1, 28, 28)
IMAGE_PAD = SLOTS - IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * IMAGE_SHAPE[2]
MNIST_TEST_MEAN = 0.1307
MNIST_TEST_STD = 0.3081

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
    img = img.reshape((1,28,28))
    img = img / 255
    img = (img - MNIST_TEST_MEAN) / MNIST_TEST_STD
    img = img.reshape(-1)
    img = np.append(img, np.array([0 for _ in range(IMAGE_PAD)]))
    inputs = {'img': img}

    name = "CryptoNets"

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
    print("'outputs.txt' save finished.")
    check_memory()

    # prediction
    outputs = outputs[0:10]
    res = outputs.index(max(outputs))
    print(img_path + " - prediction: " + str(res) + ".")



