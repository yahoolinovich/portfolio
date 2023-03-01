from django.shortcuts import render
from django.http import HttpResponse
from .forms import ImageForm
import imageio.v2 as imageio
import numpy as np
import os
from django.http import JsonResponse


syn_weights = np.loadtxt('Custom_syn_weights.txt')
syn_weights1 = np.loadtxt('Custom_syn_weights1.txt')
# Create your views here.
def index(request):
    return render(request, 'index.html')

def dev(request):
    return render(request, 'dev.html')

def renders(request):
    return render(request, '3d.html') 

def upload_image(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            result = test_img('myapp/media/images/image.png')
            os.remove(r'C:\Users\User\Downloads\split_template\Portfolio\djangerman\formedium\myapp\media\images\image.png')
            return JsonResponse({'result': result})
        else:
            print(form.errors)
    else:
        form = ImageForm()
    return render(request, 'dev.html', {'form': form})


def hidden_layer_1(input1):
    layer1 = sigmoid(np.dot(syn_weights,input1))

    return layer1


def hidden_layer_2(input2):
    layer2 = sigmoid(np.dot(syn_weights1,input2))
    return layer2


def test_img(img):
    img = imageio.imread(img, pilmode='RGBA')
    img_data = np.array(img[:, :, 3]).reshape(-1)
    img_data = (img_data / 255.0 * 0.99) + 0.01
    hidden1 = hidden_layer_1(img_data)
    hidden2 = hidden_layer_2(hidden1)
    print(hidden2)
    result = np.argmax(hidden2)
    return f'{result}'


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
