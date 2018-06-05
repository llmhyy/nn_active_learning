'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse

from scipy.misc import imsave

from driving_models import *
from utils import *
import time
import os

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(
    description='Main function for difference-inducing input generation in Driving dataset')
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
# parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
# parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=0, type=int)
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner coordinate", default=(0, 0), type=tuple)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(50, 50), type=tuple)

args = parser.parse_args()

# input image dimensions
img_rows, img_cols = 100, 100
input_shape = (img_rows, img_cols, 3)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
K.set_learning_phase(0)
model1 = Dave_orig(input_tensor=input_tensor, load_weights=True)
model2 = Dave_norminit(input_tensor=input_tensor, load_weights=True)
model3 = Dave_dropout(input_tensor=input_tensor, load_weights=True)
if args.target_model == 0:
    model = model1
elif args.target_model == 1:
    model = model2
elif args.target_model == 2:
    model = model3
# init coverage table
model_layer_dict1 = init_coverage_tables(model)

# ==============================================================================================
# start gen inputs
img_paths = image.list_pictures('./testing/center', ext='jpg')

print("begin time:",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
count = 0

for i in xrange(args.seeds):
    gen_img = preprocess_image(random.choice(img_paths))
    orig_img = gen_img.copy()

    angle1 = model.predict(gen_img)[0]
    update_coverage(gen_img, model, model_layer_dict1, args.threshold)

    print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f'
          % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2]) + bcolors.ENDC)

    # if all turning angles roughly the same
    orig_angle1 = angle1
    # layer_name1, index1 = neuron_to_cover(model_layer_dict1)

    # construct joint loss function
    # loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])
    loss1_neuron = K.mean(model.get_layer('before_prediction').output[..., 0])

    layer_output = loss1_neuron

    # for adversarial image generation
    final_loss = K.mean(layer_output)

    # we compute the gradient of the input picture wrt this loss
    grads = normalize(K.gradients(final_loss, input_tensor)[0])

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_tensor], [loss1_neuron, grads])

    # we run gradient ascent for 20 steps
    for iters in xrange(args.grad_iterations):
        loss_neuron1, grads_value = iterate([gen_img])
        if args.transformation == 'light':
            grads_value = constraint_light(grads_value)  # constraint the gradients value
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point,
                                          args.occlusion_size)  # constraint the gradients value
        elif args.transformation == 'blackout':
            grads_value = constraint_black(grads_value)  # constraint the gradients value

        gen_img += grads_value * args.step
        angle1 = model.predict(gen_img)[0]

        if angle_diverged(angle1, orig_angle1):
            update_coverage(gen_img, model, model_layer_dict1, args.threshold)
            print(bcolors.OKBLUE + 'covered neurons percentage %d neurons %.3f'
                  % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2]) + bcolors.ENDC)

            gen_img_deprocessed = draw_arrow2(deprocess_image(gen_img), orig_angle1, angle1)
            orig_img_deprocessed = draw_arrow1(deprocess_image(orig_img), orig_angle1)

            # save the result to disk
            store_path = './generated_inputs/model_' + str(args.target_model) + "/" + args.transformation + "/" + str(i)
            isExists = os.path.exists(store_path)
            if not isExists:
                os.makedirs(store_path)

            imsave(store_path + "/" + str(angle1) + '.png', gen_img_deprocessed)
            imsave(store_path + "/" + str(orig_angle1) + '_orig.png', orig_img_deprocessed)
            break

print("end time:",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print(count)
